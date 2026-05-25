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
"""Utility helpers for compiling and running EasyDeL training/eval steps.

This module hosts the lower-level building blocks used by trainers:

* Quantization helpers and straight-through-estimator emulators for
  low-precision training (``mxfp8``, ``nvfp8``, ``nf4``, ...).
* :func:`compile_trainer_step`, the thin wrapper around ``jax.jit`` /
  ``jax.pjit`` that produces sharded and (optionally) scan-friendly step
  functions.
* Pipeline-parallel scheduling utilities (``scheduled_training_step`` and
  friends) that drive MPMD schedulers.
* Generation kwarg normalization helpers used by every trainer that calls
  into the model's ``generate`` / eSurge entry points.
* Misc utilities such as ``filter_kwargs_for_callable`` for safely
  dispatching to user-supplied reward callables.
"""

from __future__ import annotations

import collections
import collections.abc
import contextlib
import dataclasses
import functools
import inspect
import time
import typing as tp
import warnings

import jax
import numpy as np
import spectrax as spx
from jax import lax
from jax import numpy as jnp
from jax import tree_util as tu
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics
from easydel.infra.sharding import MeshLike
from easydel.utils.helpers import check_bool_flag, get_logger

logger = get_logger("EasyDeL-ScheduledTrainerStep")

if tp.TYPE_CHECKING:
    from easydel.infra.etils import MpMdSchedulers

SCAN_TRAINER = check_bool_flag("SCAN_TRAINER")
FAST_COMPILE = check_bool_flag("FAST_COMPILE")
_UNSPECIFIED = object()


QuantizationMode = tp.Literal[
    "nf4",
    "affine",
    "mxfp8",
    "nvfp8",
    "mxfp4",
    "nvfp4",
]
AFFINE_SUPPORTED_BITS = frozenset({2, 3, 4, 5, 6, 7, 8})
FIXED_QUANTIZATION_BITS_BY_MODE: dict[QuantizationMode, int] = {
    "nf4": 4,
    "mxfp4": 4,
    "nvfp4": 4,
    "mxfp8": 8,
    "nvfp8": 8,
}

GENERATION_MODEL_INPUT_KEYS = (
    "inputs_embeds",
    "position_ids",
    "token_type_ids",
    "cache_position",
    "decoder_position_ids",
    "pixel_values",
    "pixel_attention_mask",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "image_grid_hws",
    "image_sizes",
    "image_max_grid_size",
    "video_max_grid_size",
    "visual_pos_masks",
    "deepstack_visual_embeds",
    "rope_deltas",
    "mm_token_type_ids",
    "image_embeds",
    "video_embeds",
    "visual_embeds",
    "image_hidden_states",
    "video_hidden_states",
    "image_features",
    "video_features",
)

SHARED_GENERATION_MODEL_INPUT_KEYS = frozenset(
    {
        "image_max_grid_size",
        "video_max_grid_size",
    }
)

GROUPED_MULTIMODAL_MODEL_INPUT_KEYS = frozenset(
    {
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "image_grid_hws",
        "image_sizes",
    }
)

PROMPT_SCORING_EXTENSION_KEYS = frozenset(
    {
        "token_type_ids",
        "mm_token_type_ids",
        "pixel_attention_mask",
        "visual_pos_masks",
    }
)

PROMPT_ONLY_SCORING_MODEL_INPUT_KEYS = frozenset(
    {
        "position_ids",
        "cache_position",
        "decoder_position_ids",
        "visual_pos_masks",
        "rope_deltas",
    }
)

_ScheduledLossFn = tp.Callable[[tp.Any, collections.abc.Mapping[str, jax.Array]], jax.Array]
_ScheduledValueAndGradFn = tp.Callable[[tp.Any, dict], tuple[jax.Array, tp.Any]]
_SCHEDULED_LOSS_ADAPTERS: dict[tuple[str, str], ScheduledLossAdapter] = {}
_SCHEDULED_AUXILIARY_CACHE: dict[tuple[int, int], tp.Callable[..., tp.Any]] = {}
_SCHEDULED_AUX_PIPELINE_EXECUTOR: list[tp.Any] = []


def _scheduled_aux_pipeline_executor() -> tp.Any:
    """Return a cached :class:`spx.MpmdPipelineExecutor` for the auxiliary (teacher/ref) forward.

    Used by :func:`cached_scheduled_auxiliary` to wavefront-overlap the per-microbatch
    forward-only auxiliary MPMD pipeline. ``use_workers=True`` runs one resident daemon worker
    per physical pipeline rank so disjoint stage submeshes actually execute concurrently --
    ``use_workers=False`` (the previous setting) had the wavefront loop ``wait_stage(...)``-block
    on each stage's device output before building the next stage's inputs, so the host could
    never get ahead and the "pipeline" ran ~serially (~196s for a forward-only pass that should
    be ~10-25s). ``dispatch_many`` still wires its deterministic ordered-transport gate through
    to the workers, so multi-controller pair-mesh collectives stay in one global launch order.
    Set ``EASYDEL_AUX_PIPELINE_INLINE=1`` to force the old inline (single-thread) executor.
    Returns ``None`` if the executor cannot be constructed (then the caller falls back to
    plain sequential ``spx.jit`` calls).
    """
    if _SCHEDULED_AUX_PIPELINE_EXECUTOR:
        return _SCHEDULED_AUX_PIPELINE_EXECUTOR[0]
    use_workers = not check_bool_flag("EASYDEL_AUX_PIPELINE_INLINE", default=False)
    executor = None
    try:
        executor = spx.MpmdPipelineExecutor(use_workers=use_workers)
        try:
            if int(jax.process_index()) == 0:
                logger.info("cached_scheduled_auxiliary: MpmdPipelineExecutor(use_workers=%s)", use_workers)
        except Exception:
            pass
    except Exception:
        try:
            executor = spx.MpmdPipelineExecutor(use_workers=False)
        except Exception:
            executor = None
    _SCHEDULED_AUX_PIPELINE_EXECUTOR.append(executor)
    return executor


_SCHED_SECTION_LOG_BUDGET = [36]


def _log_sched_section(label: str, t0: float, block_on: tp.Any = None) -> None:
    """Log how long a scheduled-step section took (wall, after a device sync on ``block_on``)."""
    if _SCHED_SECTION_LOG_BUDGET[0] <= 0:
        return
    if block_on is not None:
        try:
            jax.block_until_ready(block_on)
        except Exception:
            pass
    elapsed = time.perf_counter() - t0
    try:
        if int(jax.process_index()) != 0:
            return
    except Exception:
        return
    _SCHED_SECTION_LOG_BUDGET[0] -= 1
    logger.debug("scheduled_training_step section %-34s : %.3fs", label, elapsed)


@dataclasses.dataclass(frozen=True)
class ScheduledStepCall:
    """Frozen snapshot of one trainer step invocation, passed to scheduled-loss adapters.

    When a training step decorated with :func:`compile_trainer_step` is invoked
    under an MPMD pipeline schedule, the wrapper packages the live arguments
    of *that* call into a ``ScheduledStepCall`` and forwards it to the adapter
    registered for the underlying step function (see
    :func:`register_scheduled_loss_adapter`). The adapter uses the snapshot
    to (a) build a *trainer-specific* scalar loss closure consumed by
    ``spx.jit(..., schedule=...)`` / :func:`spx.sxvalue_and_grad`, (b) compute
    a cache key so that repeated calls with the same shape/dtype signature
    reuse the compiled scheduled loss, and (c) optionally rewrite the batch
    that flows into the scheduled loss.

    Instances are frozen and hashable-by-identity; do not mutate the captured
    mappings — the contained pytrees are still live state from the caller.

    Attributes:
        step_fn (Callable[..., Any]): The original (undecorated) trainer step
            function whose adapter is being looked up. Used purely for
            adapter-registry lookups and naming; never invoked from inside
            the adapter.
        state (EasyDeLState): The current trainer state pytree (model graphdef
            + optimizer state + step counter). This is the differentiation
            target the resulting value-and-grad runs against.
        batch (Mapping[str, jax.Array]): The mini-batch dict as passed to the
            step function (may be modified by ``ScheduledLossAdapter.prepare_batch``
            before reaching the compiled loss).
        args (tuple[Any, ...]): The positional arguments the wrapper received,
            preserved verbatim so adapters can access trailer args (e.g.
            optional reference logps).
        kwargs (Mapping[str, Any]): The keyword arguments the wrapper received,
            preserved verbatim alongside ``args``.
        bound_arguments (Mapping[str, Any]): A flat ``name -> value`` mapping
            produced by binding ``args`` / ``kwargs`` against the wrapped
            step function's signature. Use :meth:`get` for safe lookup of
            optional parameters.
        schedule (Any): The active MPMD schedule object (typically an
            ``MpMdSchedulers`` instance) under which the scheduled loss will
            be compiled. Adapters use this to specialize compilation (e.g.
            change pipeline microbatch handling).
    """

    step_fn: tp.Callable[..., tp.Any]
    state: EasyDeLState
    batch: collections.abc.Mapping[str, jax.Array]
    args: tuple[tp.Any, ...]
    kwargs: collections.abc.Mapping[str, tp.Any]
    bound_arguments: collections.abc.Mapping[str, tp.Any]
    schedule: tp.Any

    def get(self, name: str, default: tp.Any = None) -> tp.Any:
        """Look up a bound argument by name.

        Args:
            name: The argument name as it appears in the original step
                function signature.
            default: Returned when ``name`` is not present in
                ``bound_arguments``.

        Returns:
            The bound argument value or ``default``.
        """
        return self.bound_arguments.get(name, default)


@dataclasses.dataclass(frozen=True)
class ScheduledLossAdapter:
    """Trainer-specific glue between a step function and ``spx.jit(schedule=...)``.

    A ``ScheduledLossAdapter`` is registered once per *step function flavor*
    (SFT, DPO, KTO, GRPO, …) via :func:`register_scheduled_loss_adapter`, and
    is consulted by :func:`_compile_scheduled_training_step` whenever that
    step function is compiled under a non-trivial MPMD schedule.

    The adapter must satisfy three responsibilities, modelled as the three
    callable fields below:

    1. **Build a scalar loss** the SpectraX scheduler can differentiate. The
       loss is what will be compiled with ``spx.jit(loss, schedule=...)`` and
       passed to :func:`spx.sxvalue_and_grad`, so it must take exactly
       ``(state_tree, batch_dict) -> scalar`` regardless of the underlying
       trainer's richer step signature.
    2. **Produce a cache key**. The compiled scheduled value-and-grad is
       expensive; the key returned here decides when it is safe to reuse the
       cached compilation versus retrace. Include any tensor shapes/dtypes
       and trainer flags that change the loss closure.
    3. **Optionally rewrite the batch**. If the trainer needs to inject extra
       tensors (reference logps, scheduling masks, …) before the compiled
       loss sees the batch, ``prepare_batch`` returns the modified mapping;
       the original ``batch`` on :class:`ScheduledStepCall` is left untouched.

    Adapters are stored in the module-level ``_SCHEDULED_LOSS_ADAPTERS``
    registry keyed by ``(module, qualname)`` of the step function.

    Attributes:
        name (str): Short, human-readable adapter tag (``"sft"``, ``"dpo"``,
            ``"grpo"``, …). Embedded into the generated step function's
            ``__name__`` for traceability in profiles and logs.
        make_loss (Callable[[ScheduledStepCall], _ScheduledLossFn]): Factory
            that, given the live call context, returns the scalar
            ``(tree, batch) -> jax.Array`` loss closure to be JIT-compiled
            under the schedule.
        make_cache_key (Callable[[ScheduledStepCall], tuple[Any, ...]]):
            Factory that returns a hashable cache key derived from the call
            context. Two calls returning equal keys must be safe to share
            the same compiled loss.
        prepare_batch (Callable[[ScheduledStepCall], Mapping[str, jax.Array]] | None):
            Optional pre-processor invoked on every call to produce the
            mapping that flows into the compiled loss. ``None`` means the
            untouched ``ScheduledStepCall.batch`` is forwarded as-is.
    """

    name: str
    make_loss: tp.Callable[[ScheduledStepCall], _ScheduledLossFn]
    make_cache_key: tp.Callable[[ScheduledStepCall], tuple[tp.Any, ...]]
    prepare_batch: tp.Callable[[ScheduledStepCall], collections.abc.Mapping[str, jax.Array]] | None = None


@dataclasses.dataclass
class _ScheduledValueAndGradCompiler:
    """Per-step lazy compiler/cache for ``spx.jit(schedule=...)`` + ``sxvalue_and_grad``.

    One instance is created inside the closure of each scheduled training step
    produced by :func:`_compile_scheduled_training_step`. On every call the
    instance asks its adapter for a cache key built from the current
    :class:`ScheduledStepCall`; if that key matches the previously seen one,
    the cached compiled value-and-grad function is reused directly. Otherwise
    the compiler:

    1. Asks the adapter to materialise the scalar loss closure.
    2. Compiles it through ``spx.jit`` with the step's mesh, MPMD schedule,
       and ``batch_argnums`` (so the scheduler knows which positional argument
       carries the per-microbatch tensors).
    3. Wraps the compiled function with :func:`spx.sxvalue_and_grad` against
       the state tree (``argnums=0``) and stores it for future reuse.

    The class is mutable on purpose — mutation is the cache update. There is
    no thread safety here because each compiler lives behind a single
    ``scheduled_training_step`` closure that is invoked sequentially by the
    trainer loop.

    Attributes:
        mesh (MeshLike): Spectrax/JAX mesh the compiled loss runs on.
            Forwarded to ``spx.jit``.
        schedule (Any): MPMD schedule object passed to ``spx.jit(..., schedule=...)``.
        batch_argnums (int | Sequence[int] | None): Positional indices of the
            scheduled-loss arguments that carry per-microbatch data. ``None``
            disables microbatch slicing.
        adapter (ScheduledLossAdapter): Trainer-specific adapter used to build
            both the loss closure and the cache key.
        cached_key (tuple[Any, ...] | None): Last cache key returned by
            ``adapter.make_cache_key``. ``None`` until the first compilation.
        cached_value_and_grad (_ScheduledValueAndGradFn | None): Last
            compiled ``(tree, batch) -> (loss, gradients)`` function. ``None``
            until the first compilation; reused while ``cached_key`` matches.
    """

    mesh: MeshLike
    schedule: tp.Any
    batch_argnums: int | tp.Sequence[int] | None
    adapter: ScheduledLossAdapter
    cached_key: tuple[tp.Any, ...] | None = None
    cached_value_and_grad: _ScheduledValueAndGradFn | None = None

    def get(self, call: ScheduledStepCall) -> _ScheduledValueAndGradFn:
        """Return a cached scheduled value-and-grad callable for ``call``.

        On a cache miss, builds a fresh ``spx.jit``-compiled loss with the
        configured schedule and wraps it in :func:`spx.sxvalue_and_grad`.
        Subsequent calls with the same adapter cache key reuse the existing
        compiled callable.

        Args:
            call: The current scheduled step call context.

        Returns:
            A function ``(tree, batch) -> (loss, gradients)`` ready to be
            applied by the trainer.
        """
        key = self.adapter.make_cache_key(call)
        if self.cached_value_and_grad is not None and self.cached_key == key:
            return self.cached_value_and_grad

        loss_fn = self.adapter.make_loss(call)
        scheduled_loss = spx.jit(
            loss_fn,
            mesh=self.mesh,
            schedule=self.schedule,
            static_argnums=(),
            batch_argnums=self.batch_argnums,
        )
        scheduled_value_and_grad = spx.sxvalue_and_grad(scheduled_loss, argnums=0)

        def value_and_grad(tree, batch):
            """Run the scheduled value-and-grad and unwrap the gradient tuple.

            Args:
                tree: The state pytree to differentiate against.
                batch: The minibatch dictionary forwarded to the loss.

            Returns:
                A ``(loss, gradients)`` tuple where ``gradients`` matches
                the structure of ``tree``.
            """
            loss, (gradients,) = scheduled_value_and_grad(tree, batch)
            return loss, gradients

        self.cached_key = key
        self.cached_value_and_grad = value_and_grad
        return value_and_grad


def _make_eformer_stage_local_apply_fn(
    tx: tp.Any,
) -> tp.Callable[..., None]:
    """Build a SpectraX-compatible ``apply_fn`` from an eFormer stage-local optimizer.

    The returned callable matches the contract ``sxvalue_and_grad_and_apply`` expects:
    ``apply_fn(rank, *, grad_accums, state)`` mutates ``state["new_params_buf"][rank]`` and
    ``state["new_opt_state_buf"][rank]`` with the optimizer-updated full params/opt-state
    trees (only the leaves owned by ``rank`` are actually updated; leaves owned by other
    ranks are passed through unchanged, so the default last-write-wins assembler at the
    runtime exit yields a correct merged tree).

    The work-per-rank goes through the optimizer's ``apply_gradients_stage_local`` method
    (eFormer's stage-local AdamW path -- the batched per-submesh kernel lives directly in
    ``eformer.optimizers._stage_local._apply_adamw_stage_local`` and is the default
    implementation, not a monkey-patch). That kernel does the right thing when handed a
    *sparse* gradient tree (leaves owned by other ranks set to ``None``): the per-submesh
    grouping naturally collapses to one group for the rank's submesh, and ``None``-grad
    leaves are carried through unchanged.

    Args:
        tx: The optimizer state's ``tx`` attribute -- must expose
            ``apply_gradients_stage_local(params, grads, opt_state, learning_rate_fn,
            delete_grads)``. eFormer's chained AdamW already does.

    Returns:
        A callable suitable for passing as ``apply_fn`` to
        :func:`spectrax.sxvalue_and_grad_and_apply`.

    Raises:
        RuntimeError: At call time, if the optimizer does not implement
            ``apply_gradients_stage_local``.
    """
    apply_stage_local = getattr(tx, "apply_gradients_stage_local", None)
    if not callable(apply_stage_local):
        raise RuntimeError(
            "Fused stage-local apply requires an optimizer with "
            "`apply_gradients_stage_local` (use an eFormer chained AdamW)."
        )

    is_leaf_none = lambda x: x is None  # noqa: E731

    def apply_fn(rank: int, *, grad_accums: dict, state: dict) -> None:
        """Per-rank stage-local optimizer apply.

        Builds a sparse gradient tree where only leaves owned by ``rank`` carry their
        gradient (other leaves are None), then calls the optimizer's
        ``apply_gradients_stage_local`` with the full params / sparse grads / full
        opt_state. The kernel updates only the leaves with non-None grads (i.e. this
        rank's leaves), leaving everything else identity-passed-through. The result
        is the full new_params / new_opt_state tree with *only this rank's leaves
        updated*; the assembler in :func:`_assemble_fused_apply_outputs` walks every
        rank's tree and merges per-leaf-by-owner to produce the correct global
        new_params / new_opt_state.

        Args:
            rank: Physical pipeline rank this apply unit owns.
            grad_accums: SpectraX dispatcher's ``flat_idx -> accumulated grad``
                map. May contain entries from other ranks (if they happened to
                finish bwd before this rank) -- we filter by ``leaf_stage_owners``.
            state: Mutable apply context dict from
                ``sxvalue_and_grad_and_apply``. Reads ``params``, ``opt_state``,
                ``learning_rate_fn``, ``leaf_stage_owners``. Writes
                ``new_params_buf[rank]`` and ``new_opt_state_buf[rank]``.
        """
        params = state["params"]
        opt_state = state["opt_state"]
        learning_rate_fn = state["learning_rate_fn"]
        leaf_stage_owners: dict = state["leaf_stage_owners"]
        treedef = tu.tree_structure(params, is_leaf=is_leaf_none)
        p_flat = tu.tree_leaves(params, is_leaf=is_leaf_none)
        sparse_g_flat: list = []
        for flat_idx in range(len(p_flat)):
            owner = leaf_stage_owners.get(flat_idx)
            if owner is None or owner == rank:
                sparse_g_flat.append(grad_accums.get(flat_idx))
            else:
                sparse_g_flat.append(None)
        sparse_grads = tu.tree_unflatten(treedef, sparse_g_flat)
        new_params, new_opt_state = apply_stage_local(
            params=params,
            grads=sparse_grads,
            opt_state=opt_state,
            learning_rate_fn=learning_rate_fn,
            delete_grads=False,
        )
        state["new_params_buf"][rank] = new_params
        state["new_opt_state_buf"][rank] = new_opt_state

    return apply_fn


def _assemble_fused_apply_outputs(apply_context: dict) -> tuple[tp.Any, tp.Any]:
    """Merge per-rank fused apply outputs into a single new_params / new_opt_state.

    Each rank's apply_fn writes the full new_params / new_opt_state tree to its
    buffer, but only the leaves owned by that rank are actually updated -- the
    other leaves are the original input values (eFormer's stage-local kernel
    skips leaves with None grads). The correct merged tree picks, for each leaf
    position, the array from the buffer of the rank that owns that leaf.

    Args:
        apply_context: The mutable context dict from
            :func:`spectrax.sxvalue_and_grad_and_apply`. Reads
            ``new_params_buf``, ``new_opt_state_buf``, ``leaf_stage_owners``,
            ``params`` (treedef template), ``opt_state`` (treedef template).

    Returns:
        ``(new_params, new_opt_state)`` -- the merged trees.

    Raises:
        RuntimeError: If no rank wrote into the buffers (apply_fn never fired).
    """
    new_params_buf = apply_context["new_params_buf"]
    new_opt_state_buf = apply_context["new_opt_state_buf"]
    leaf_stage_owners: dict = apply_context["leaf_stage_owners"]
    if not new_params_buf or not new_opt_state_buf:
        raise RuntimeError(
            "Fused stage-local apply produced empty per-rank output buffers. "
            "This indicates the apply units never fired -- check that the schedule "
            "has enough physical ranks to host the model and that apply_jits got "
            "populated for every rank."
        )
    is_leaf_none = lambda x: x is None  # noqa: E731

    template_rank = next(iter(new_params_buf))
    template_params = new_params_buf[template_rank]
    p_treedef = tu.tree_structure(template_params, is_leaf=is_leaf_none)
    p_flats = {r: tu.tree_leaves(new_params_buf[r], is_leaf=is_leaf_none) for r in new_params_buf}
    n_leaves = len(p_flats[template_rank])
    merged_p: list = []
    for i in range(n_leaves):
        owner = leaf_stage_owners.get(i, template_rank)
        source_rank = owner if owner in p_flats else template_rank
        merged_p.append(p_flats[source_rank][i])
    new_params = tu.tree_unflatten(p_treedef, merged_p)

    template_opt_state = new_opt_state_buf[template_rank]
    os_treedef = tu.tree_structure(template_opt_state, is_leaf=is_leaf_none)
    os_flats = {r: tu.tree_leaves(new_opt_state_buf[r], is_leaf=is_leaf_none) for r in new_opt_state_buf}
    template_os_flat = os_flats[template_rank]
    n_os_leaves = len(template_os_flat)
    merged_os: list = []
    for i in range(n_os_leaves):
        leaf_template = template_os_flat[i]
        owner_rank = None
        sharding = getattr(leaf_template, "sharding", None)
        if sharding is not None:
            try:
                template_devices = sharding.device_set
            except Exception:
                template_devices = None
            if template_devices is not None:
                rank_submeshes = apply_context.get("rank_submeshes", ())
                for r in new_opt_state_buf:
                    if r >= len(rank_submeshes):
                        continue
                    submesh = rank_submeshes[r]
                    devices = getattr(submesh, "devices", None)
                    if devices is None:
                        continue
                    device_set = set(devices.flat) if hasattr(devices, "flat") else set(devices)
                    if device_set == template_devices:
                        owner_rank = r
                        break
        if owner_rank is None:
            owner_rank = template_rank
        source_rank = owner_rank if owner_rank in os_flats else template_rank
        merged_os.append(os_flats[source_rank][i])
    new_opt_state = tu.tree_unflatten(os_treedef, merged_os)

    return new_params, new_opt_state


@dataclasses.dataclass
class _ScheduledValueAndGradAndApplyCompiler:
    """Per-step lazy compiler for ``spx.sxvalue_and_grad_and_apply``.

    Mirror of :class:`_ScheduledValueAndGradCompiler` but caches the fused
    value-and-grad-and-apply callable instead of the plain value-and-grad.
    Caller invokes the returned callable as
    ``fused_step(tree, batch, opt_state, learning_rate_fn, apply_fn)``
    and gets back ``(loss, new_tree, new_opt_state)`` -- the optimizer apply has
    already run inside the MPMD schedule as per-rank APPLY units.

    Attributes:
        mesh: The SpectraX/JAX mesh the compiled loss runs on.
        schedule: The MPMD schedule.
        batch_argnums: Positional indices of the per-microbatch arguments.
        adapter: The trainer-specific :class:`ScheduledLossAdapter`.
        cached_key: Last adapter cache key (``None`` until first compile).
        cached_fused: Last compiled fused callable.
    """

    mesh: MeshLike
    schedule: tp.Any
    batch_argnums: int | tp.Sequence[int] | None
    adapter: ScheduledLossAdapter
    cached_key: tuple[tp.Any, ...] | None = None
    cached_fused: tp.Callable[..., tp.Any] | None = None

    def get(self, call: ScheduledStepCall) -> tp.Callable[..., tp.Any]:
        """Return a cached fused value-and-grad-and-apply callable for ``call``.

        On cache miss, compiles a fresh ``spx.jit``-wrapped loss with the
        configured schedule and wraps it in
        :func:`spectrax.sxvalue_and_grad_and_apply`.

        Args:
            call: Current scheduled step call context.

        Returns:
            A callable ``(tree, batch, opt_state, learning_rate_fn, apply_fn)
            -> (loss, new_tree, new_opt_state)``.
        """
        key = self.adapter.make_cache_key(call)
        if self.cached_fused is not None and self.cached_key == key:
            return self.cached_fused

        loss_fn = self.adapter.make_loss(call)
        scheduled_loss = spx.jit(
            loss_fn,
            mesh=self.mesh,
            schedule=self.schedule,
            static_argnums=(),
            batch_argnums=self.batch_argnums,
        )
        scheduled_vga = spx.sxvalue_and_grad_and_apply(scheduled_loss, argnums=0)

        def fused_step(tree, batch, opt_state, learning_rate_fn, apply_fn):
            """Run the scheduled fused value-and-grad-and-apply.

            Args:
                tree: The state pytree to differentiate against and update.
                batch: Per-microbatch input dict.
                opt_state: Optimizer state pytree matching ``tree``.
                learning_rate_fn: Optional optax schedule callback.
                apply_fn: Per-rank apply callable (see
                    :func:`_make_eformer_stage_local_apply_fn`).

            Returns:
                ``(loss, new_tree, new_opt_state)``.
            """
            return scheduled_vga(
                tree,
                batch,
                apply_fn=apply_fn,
                opt_state=opt_state,
                learning_rate_fn=learning_rate_fn,
                assemble_outputs=_assemble_fused_apply_outputs,
            )

        self.cached_key = key
        self.cached_fused = fused_step
        return fused_step


def filter_kwargs_for_callable(
    callable_obj: tp.Callable[..., tp.Any],
    kwargs: collections.abc.Mapping[str, tp.Any],
) -> dict[str, tp.Any]:
    """Filter kwargs so only parameters accepted by ``callable_obj`` are forwarded.

    This prevents runtime failures when dataset batches carry auxiliary metadata
    fields (for example preference scores) that a model forward signature does
    not accept.
    """
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)

    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(kwargs)

    accepted_keys = set(parameters.keys())
    return {key: value for key, value in kwargs.items() if key in accepted_keys}


def sanitize_model_call_kwargs(kwargs: collections.abc.Mapping[str, tp.Any]) -> dict[str, tp.Any]:
    """Normalize model call kwargs to avoid known incompatible combinations.

    Causal LM forwards generally accept either ``input_ids`` or ``inputs_embeds``,
    but not both at the same time. Prefer token IDs when both are present.
    """
    normalized_kwargs = dict(kwargs)
    if normalized_kwargs.get("input_ids", None) is not None and normalized_kwargs.get("inputs_embeds", None) is not None:
        normalized_kwargs.pop("inputs_embeds", None)
    return normalized_kwargs


def register_scheduled_loss_adapter(
    step_fn: tp.Callable[..., tp.Any],
    adapter: ScheduledLossAdapter,
) -> tp.Callable[..., tp.Any]:
    """Bind a :class:`ScheduledLossAdapter` to a trainer step function.

    The registry maintained here lets :func:`_compile_scheduled_training_step`
    discover, given a raw step function, how to build the scalar loss that
    SpectraX's scheduled VJP needs. Concretely the adapter is the
    *trainer-specific* piece (DPO/KTO/PPO/SFT/...) that knows how to:

    * project the live ``state``/``batch`` into a
      ``(tree, batch) -> jax.Array`` scalar loss closure,
    * build a hashable cache key so equivalent calls reuse the compiled
      ``spx.jit(..., schedule=...)`` artifact, and
    * optionally pre-process the batch before it reaches the compiled loss.

    Everything else — gradient accumulation, scheduled VJP, stage-local
    gradient placement, and the optimizer update — stays in the shared
    pipeline-parallel path and is *not* the adapter's concern.

    Registration writes to two locations so lookups via
    :func:`get_scheduled_loss_adapter` succeed regardless of how the caller
    holds the function:

    1. The module-level ``_SCHEDULED_LOSS_ADAPTERS`` dict, keyed by
       ``(module, qualname)`` of ``step_fn``. This survives even when
       ``step_fn`` is wrapped or copied later.
    2. The function attribute ``step_fn.__easydel_scheduled_loss_adapter__``,
       a fast-path direct pointer used when the caller still has the
       original object.

    The function is intentionally usable as a decorator factory (returns
    ``step_fn`` unchanged) so trainers can write
    ``register_scheduled_loss_adapter(step_fn=step_fn, adapter=adapter)`` at
    module import time.

    Args:
        step_fn (Callable[..., Any]): The trainer's raw, unwrapped step
            function (e.g. the SFT ``training_step``, DPO ``training_step``,
            …) that will later be compiled under an MPMD schedule. Used only
            for registry-key derivation and attribute attachment; never
            invoked by this function.
        adapter (ScheduledLossAdapter): The trainer-specific adapter that
            knows how to materialise a scalar loss / cache key / batch
            override from a live :class:`ScheduledStepCall`. Stored by
            reference, so the caller must not mutate it after registration.

    Returns:
        Callable[..., Any]: The same ``step_fn`` object passed in,
        unmodified except for the freshly-attached
        ``__easydel_scheduled_loss_adapter__`` attribute. Returning
        ``step_fn`` lets this function be chained or used as a decorator.
    """

    _SCHEDULED_LOSS_ADAPTERS[_scheduled_step_key(step_fn)] = adapter
    step_fn.__easydel_scheduled_loss_adapter__ = adapter
    return step_fn


def get_scheduled_loss_adapter(fn: tp.Callable[..., tp.Any]) -> ScheduledLossAdapter | None:
    """Return the :class:`ScheduledLossAdapter` registered for ``fn``, if any.

    Adapters can be attached either directly via the
    ``__easydel_scheduled_loss_adapter__`` attribute or through
    :func:`register_scheduled_loss_adapter`.  Both lookup paths are
    consulted.

    Args:
        fn: A trainer step callable.

    Returns:
        The associated adapter, or ``None`` when none has been registered.
    """
    adapter = getattr(fn, "__easydel_scheduled_loss_adapter__", None)
    if adapter is not None:
        return adapter
    return _SCHEDULED_LOSS_ADAPTERS.get(_scheduled_step_key(fn))


def scheduled_cache_token(value: tp.Any) -> tp.Hashable:
    """Return a stable token for scheduled-loss cache keys."""

    try:
        hash(value)
    except TypeError:
        return id(value)
    return value


def scheduled_loss_cache_key(
    call: ScheduledStepCall,
    *,
    value_fields: collections.abc.Iterable[str] = (),
    object_fields: collections.abc.Iterable[str] = (),
    include_graph: bool = True,
) -> tuple[tp.Any, ...]:
    """Build a cache key for scheduled scalar-loss compilation.

    ``value_fields`` are hashed by value when possible, while
    ``object_fields`` are keyed by identity.  This keeps trainer adapters from
    open-coding the same graph/config/function identity tuple.
    """

    pieces: list[tp.Any] = []
    if include_graph:
        pieces.extend((id(call.state.graphdef), id(call.state.graphother)))
    pieces.extend(scheduled_cache_token(call.get(name)) for name in value_fields)
    pieces.extend(id(call.get(name)) for name in object_fields)
    return tuple(pieces)


def _sync_schedule_config(config: tp.Any, schedule: tp.Any, seen: set[int]) -> None:
    """Keep a config and its nested sub-configs aligned with the runtime schedule."""
    if config is None:
        return
    config_id = id(config)
    if config_id in seen:
        return
    seen.add(config_id)

    virtual_stages = getattr(schedule, "virtual_stages_per_rank", None)
    if callable(virtual_stages) and hasattr(config, "pipeline_virtual_stages"):
        config.pipeline_virtual_stages = int(virtual_stages())
    stage_layout = getattr(schedule, "stage_layout", None)
    if stage_layout is not None and hasattr(config, "pipeline_stage_layout"):
        config.pipeline_stage_layout = stage_layout

    for attr_name in ("text_config", "vision_config", "encoder_config", "decoder_config"):
        _sync_schedule_config(getattr(config, attr_name, None), schedule, seen)


def sync_module_schedule_config(module: tp.Any, schedule: tp.Any) -> None:
    """Keep model-side PP marker generation in sync with the runtime schedule."""
    _sync_schedule_config(getattr(module, "config", None), schedule, set())

    for attr_name in ("model", "base_model", "language_model", "visual"):
        child = getattr(module, attr_name, None)
        _sync_schedule_config(getattr(child, "config", None), schedule, set())


def _sync_physical_stage_config(
    config: tp.Any,
    seen: set[int],
    changes: list[tuple[tp.Any, tp.Any]],
) -> None:
    """Keep auxiliary regular MPMD forwards on physical PP stage markers."""
    if config is None:
        return
    config_id = id(config)
    if config_id in seen:
        return
    seen.add(config_id)

    if hasattr(config, "pipeline_virtual_stages"):
        changes.append((config, config.pipeline_virtual_stages))
        config.pipeline_virtual_stages = 1

    for attr_name in ("text_config", "vision_config", "encoder_config", "decoder_config"):
        _sync_physical_stage_config(getattr(config, attr_name, None), seen, changes)


@contextlib.contextmanager
def sync_module_physical_stage_config(module: tp.Any) -> tp.Iterator[None]:
    """Temporarily configure auxiliary forwards for physical PP rank markers.

    Scheduled train losses use ``sync_module_schedule_config`` so virtual-stage
    schedules emit ``pp * V`` logical markers. Auxiliary forwards compiled as a
    regular ``spx.jit`` do not pass ``schedule=...`` and must therefore emit
    exactly ``pp`` physical-rank markers. The original config values are
    restored after the auxiliary call.
    """
    changes: list[tuple[tp.Any, tp.Any]] = []
    _sync_physical_stage_config(getattr(module, "config", None), set(), changes)

    for attr_name in ("model", "base_model", "language_model", "visual"):
        child = getattr(module, attr_name, None)
        _sync_physical_stage_config(getattr(child, "config", None), set(), changes)
    try:
        yield
    finally:
        for config, old_value in reversed(changes):
            config.pipeline_virtual_stages = old_value


def bind_scheduled_module(
    call: ScheduledStepCall,
    tree: tp.Any,
    *,
    straight_through_field: str = "straight_through_emulator",
) -> tp.Any:
    """Merge a scheduled trainable tree and sync its PP marker config.

    Materialises the model module from the scheduled-step state tree,
    applying an optional straight-through quantization emulator and
    synchronising the resulting module's pipeline-stage configuration
    with the active MPMD schedule so subsequent stage markers match.

    Args:
        call: Live :class:`ScheduledStepCall` for this scheduled step.
        tree: Differentiable parameter pytree to merge into ``call.state``.
        straight_through_field: Name of the bound argument that, when
            present and non-``None``, is applied to ``tree`` as a
            quantization STE before merging.

    Returns:
        The materialised model module ready for the scheduled forward.
    """

    straight_through = call.get(straight_through_field)
    if straight_through is not None:
        tree = straight_through(tree)
    module = call.state.merge(tree)
    sync_module_schedule_config(module, call.schedule)
    return module


def constrain_scheduled_batch(
    module: tp.Any,
    batch: collections.abc.Mapping[str, tp.Any],
    partition_spec: tp.Any,
) -> dict[str, tp.Any]:
    """Apply the standard EasyDeL scheduled batch sharding constraint.

    Forwards every leaf of ``batch`` through
    ``spx.with_sharding_constraint(..., ignore_mpmd=True)`` so the
    scheduled VJP path can see the intended data-parallel partitioning
    even when the MPMD scheduler would otherwise rewrite the constraint.

    Args:
        module: The materialised model module (its ``mesh`` attribute
            is used as the constraint mesh).
        batch: Input batch dict.
        partition_spec: The ``PartitionSpec`` to apply per leaf.

    Returns:
        A new dict with the same keys as ``batch`` and constrained
        sharding annotations on every leaf.
    """

    return tp.cast(
        dict[str, tp.Any],
        spx.with_sharding_constraint(
            dict(batch),
            partition_spec,
            mesh=module.mesh,
            ignore_mpmd=True,
        ),
    )


def _aux_leading_batch_size(value: tp.Any) -> int | None:
    """Return the leading-axis size of the first array leaf in ``value``, if any."""

    for leaf in tu.tree_leaves(value):
        shape = getattr(leaf, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
    return None


def _aux_slice_tree_leading_axis(value: tp.Any, start: int, size: int, full: int) -> tp.Any:
    """Slice every array leaf of ``value`` whose leading axis equals ``full`` to ``[start:start+size]``."""

    def _slice(leaf: tp.Any) -> tp.Any:
        """Slice a single pytree leaf along axis 0 when it carries the full leading dim.

        Args:
            leaf: A pytree leaf (typically a JAX array).

        Returns:
            ``lax.slice_in_dim(leaf, start, start + size, axis=0)`` when
            ``leaf.shape[0] == full``; otherwise ``leaf`` unchanged.
        """
        shape = getattr(leaf, "shape", None)
        if shape is None or len(shape) == 0 or int(shape[0]) != full:
            return leaf
        return lax.slice_in_dim(leaf, start, start + size, axis=0)

    return tu.tree_map(_slice, value)


def _aux_concat_microbatch_outputs(microbatch_outputs: list[tp.Any], chunk: int) -> tp.Any:
    """Concatenate per-microbatch output trees along axis 0 for leaves of leading size ``chunk``.

    Each microbatch already produced a ``(chunk, ...)`` shard living on its stage submesh; the
    concatenation stitches them back into a ``(microbatches * chunk, ...)`` array that stays sharded
    over the same submesh (axis-0 inputs are sharded over the data axis, so the concat output is too).
    Leaves whose leading axis is not ``chunk`` (scalars, metadata) are passed through from the first
    microbatch unchanged.
    """

    def _concat(*leaves: tp.Any) -> tp.Any:
        """Concatenate aligned pytree leaves along axis 0 when they carry the chunk dim.

        Args:
            *leaves: One leaf from each microbatch's output tree at the same
                pytree position.

        Returns:
            ``jnp.concatenate(leaves, axis=0)`` for leaves whose leading
            axis equals ``chunk``; otherwise ``leaves[0]`` (treated as a
            shared scalar/metadata leaf).
        """
        first = leaves[0]
        shape = getattr(first, "shape", None)
        if shape is None or len(shape) == 0 or int(shape[0]) != chunk:
            return first
        return jnp.concatenate(leaves, axis=0)

    return tu.tree_map(_concat, *microbatch_outputs)


def cached_scheduled_auxiliary(
    fn: tp.Callable[..., tp.Any],
    mesh: MeshLike,
    *,
    microbatches: int = 1,
    batch_argnums: int | tp.Sequence[int] = 1,
) -> tp.Callable[..., tp.Any]:
    """Return a cached regular ``spx.jit`` for non-gradient auxiliary (teacher/reference) forwards.

    When ``microbatches > 1`` the returned wrapper runs the compiled MPMD forward once per
    schedule-sized leading-axis chunk of the batched positional argument(s) named by
    ``batch_argnums`` and concatenates the real array outputs. This mirrors the
    per-microbatch contract of the scheduled training step: the auxiliary forward is a *true*
    MPMD pipeline (the same per-rank stages, dispatched per microbatch), and no full-batch
    activation or hidden state is ever materialized — the largest live tensor inside the
    forward is ``(global_batch / microbatches, ...)``, and the reassembled output stays
    sharded over its producing stage submesh.

    ``microbatches <= 1`` (no MPMD schedule, or a 1-microbatch schedule) keeps the plain
    single-call ``spx.jit`` behavior.
    """

    batch_argnums_t: tuple[int, ...]
    if isinstance(batch_argnums, int):
        batch_argnums_t = (batch_argnums,)
    else:
        batch_argnums_t = tuple(int(i) for i in batch_argnums)
    microbatches = max(1, int(microbatches))

    key = (id(fn), id(mesh), microbatches, batch_argnums_t)
    cached = _SCHEDULED_AUXILIARY_CACHE.get(key)
    if cached is not None:
        return cached

    compiled = spx.jit(fn, mesh=mesh)
    if microbatches <= 1 or not batch_argnums_t:
        _SCHEDULED_AUXILIARY_CACHE[key] = compiled
        return compiled

    @functools.wraps(fn)
    def _microbatched(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Run the compiled forward once per microbatch and reassemble outputs.

        Slices each batched positional argument named by ``batch_argnums_t``
        along axis 0 into ``microbatches`` equal chunks, dispatches the
        ``spx.jit``-compiled forward per chunk (preferring the cached
        :class:`spx.MpmdPipelineExecutor` for wavefront overlap when
        available, otherwise sequential ``spx.jit`` calls), and concatenates
        per-microbatch real-array outputs along axis 0 with
        :func:`_aux_concat_microbatch_outputs`.

        Args:
            *args: Positional arguments forwarded to the compiled forward.
                Entries at indices in ``batch_argnums_t`` are sliced
                per-microbatch; all others are passed through unchanged.
            **kwargs: Keyword arguments forwarded verbatim. Note: the
                wavefront ``dispatch_many`` path is only used when there
                are no kwargs.

        Returns:
            The reassembled forward output with the same pytree structure
            as a single full-batch call.

        Raises:
            ValueError: If ``batch_argnums`` references positions outside
                ``args``, if batched arguments disagree on leading size,
                or if the batch size is not divisible by ``microbatches``.
        """
        batch_sizes: list[int] = []
        for argnum in batch_argnums_t:
            if argnum >= len(args):
                raise ValueError(
                    f"cached_scheduled_auxiliary: batch_argnums contains {argnum} but only "
                    f"{len(args)} positional argument(s) were passed."
                )
            size = _aux_leading_batch_size(args[argnum])
            if size is not None:
                batch_sizes.append(size)
        if not batch_sizes:
            return compiled(*args, **kwargs)
        batch_size = batch_sizes[0]
        if any(s != batch_size for s in batch_sizes):
            raise ValueError(
                f"cached_scheduled_auxiliary: batched arguments must share a leading size; got {tuple(batch_sizes)}."
            )
        if batch_size % microbatches != 0:
            raise ValueError(
                f"cached_scheduled_auxiliary: batch size {batch_size} is not divisible by microbatches {microbatches}."
            )
        chunk = batch_size // microbatches
        arg_batches: list[tuple[tp.Any, ...]] = []
        for mb in range(microbatches):
            start = mb * chunk
            sliced = list(args)
            for argnum in batch_argnums_t:
                sliced[argnum] = _aux_slice_tree_leading_axis(sliced[argnum], start, chunk, batch_size)
            arg_batches.append(tuple(sliced))
        microbatch_outputs: list[tp.Any] | None = None
        used_wavefront = False
        executor = None
        if not kwargs:
            executor = _scheduled_aux_pipeline_executor()
            if executor is not None:
                try:
                    microbatch_outputs = list(executor.dispatch_many(compiled, arg_batches))
                    used_wavefront = True
                except Exception as exc:
                    if _SCHED_SECTION_LOG_BUDGET[0] > 0:
                        logger.info(
                            "cached_scheduled_auxiliary: dispatch_many unavailable (%s); using sequential calls", exc
                        )
                    microbatch_outputs = None
        if microbatch_outputs is None:
            microbatch_outputs = [compiled(*ab, **kwargs) for ab in arg_batches]
        if _SCHED_SECTION_LOG_BUDGET[0] > 0:
            try:
                if int(jax.process_index()) == 0:
                    logger.info(
                        "cached_scheduled_auxiliary: %d microbatch(es) via %s",
                        microbatches,
                        "MpmdPipelineExecutor.dispatch_many (wavefront)"
                        if used_wavefront
                        else "sequential spx.jit calls",
                    )
                    if used_wavefront:
                        st = getattr(executor, "last_stats", None)
                        if st is not None:
                            logger.info(
                                "  dispatch_many stats: stage_launches=%s queue_wait=%.2fs dispatch=%.2fs "
                                "submit=%.2fs prepare=%.2fs assemble=%.2fs | per-stage assemble_ms=%s execute_ms=%s submit_ms=%s",
                                getattr(st, "stage_launches", None),
                                getattr(st, "queue_wait_time", 0.0) or 0.0,
                                getattr(st, "stage_dispatch_time", 0.0) or 0.0,
                                getattr(st, "submit_time", 0.0) or 0.0,
                                getattr(st, "prepare_time", 0.0) or 0.0,
                                getattr(st, "assemble_time", 0.0) or 0.0,
                                [round(x) for x in (getattr(st, "stage_assemble_times_ms", ()) or ())],
                                [round(x) for x in (getattr(st, "stage_execute_times_ms", ()) or ())],
                                [round(x) for x in (getattr(st, "stage_submit_times_ms", ()) or ())],
                            )
            except Exception:
                pass
        return _aux_concat_microbatch_outputs(microbatch_outputs, chunk)

    _SCHEDULED_AUXILIARY_CACHE[key] = _microbatched
    return _microbatched


def stop_gradient_tree(value: tp.Any) -> tp.Any:
    """Stop gradients for array leaves while preserving non-array metadata.

    Applies :func:`jax.lax.stop_gradient` to each leaf that exposes a
    ``shape`` attribute (i.e. JAX arrays), leaving Python scalars,
    strings, and other static metadata untouched. Used by scheduled
    reference forwards so that frozen-teacher outputs do not leak
    gradients back into the policy graph.

    Args:
        value: Arbitrary pytree.

    Returns:
        A pytree with the same structure as ``value`` and gradients
        cut on every array leaf.
    """

    return jax.tree_util.tree_map(
        lambda leaf: jax.lax.stop_gradient(leaf) if hasattr(leaf, "shape") else leaf,
        value,
    )


def prepare_scheduled_reference_outputs(
    call: ScheduledStepCall,
    *,
    reference_state_field: str,
    forward_field: str,
    output_to_batch: collections.abc.Mapping[str, str],
    partition_spec_field: str = "partition_spec",
    skip_field: str | None = None,
    missing_error: str | None = None,
) -> dict[str, tp.Any]:
    """Precompute reference-model outputs before the scheduled VJP.

    Preference/RL trainers often need frozen reference log-probs.  Computing
    those inside the policy scheduled loss would trace two model forwards into a
    single PP graph, so this helper runs the reference forward once as a regular
    auxiliary JIT and appends its requested outputs to the batch.
    """

    batch = dict(call.batch)
    if skip_field is not None and bool(call.get(skip_field, False)):
        return batch
    if all(batch_key in batch for batch_key in output_to_batch.values()):
        return batch

    reference_state = call.get(reference_state_field)
    forward_fn = call.get(forward_field)
    if reference_state is None or forward_fn is None:
        raise RuntimeError(
            missing_error or f"scheduled MPMD training requires {reference_state_field!r} and {forward_field!r}."
        )

    ref_model = reference_state.model
    ref_model.eval()
    with sync_module_physical_stage_config(ref_model):
        constrained_batch = spx.with_sharding_constraint(
            batch,
            call.get(partition_spec_field),
            mesh=ref_model.mesh,
            ignore_mpmd=True,
        )
        ref_forward = cached_scheduled_auxiliary(
            forward_fn,
            ref_model.mesh,
            microbatches=getattr(call.schedule, "microbatches", 1),
            batch_argnums=1,
        )
        ref_out = stop_gradient_tree(ref_forward(ref_model, constrained_batch))
    for output_key, batch_key in output_to_batch.items():
        batch[batch_key] = ref_out[output_key]
    return batch


def normalize_generation_model_kwargs(
    kwargs: collections.abc.Mapping[str, tp.Any] | None,
    *,
    model_callable: tp.Callable[..., tp.Any] | None = None,
) -> dict[str, tp.Any]:
    """Normalize model-side generation kwargs to a stable key set.

    Generation JITs work best when auxiliary model inputs use a fixed pytree
    structure. This helper keeps only known model input keys, filters them
    against the model forward signature when available, and fills missing keys
    with ``None`` so callers can safely pass the result into cached compiled
    functions.

    Args:
        kwargs: Raw mapping of model keyword arguments. May be ``None``.
        model_callable: Optional model forward callable used to filter keys
            against its signature.

    Returns:
        dict: Normalized dictionary with all ``GENERATION_MODEL_INPUT_KEYS``
            present (missing ones set to ``None``).
    """

    normalized = {key: None for key in GENERATION_MODEL_INPUT_KEYS}
    if not kwargs:
        return normalized

    extracted = {key: value for key, value in kwargs.items() if key in normalized and value is not None}
    if model_callable is not None:
        extracted = filter_kwargs_for_callable(model_callable, extracted)
    normalized.update(extracted)
    return normalized


def compact_generation_model_kwargs(kwargs: collections.abc.Mapping[str, tp.Any] | None) -> dict[str, tp.Any]:
    """Drop ``None`` leaves from normalized generation model kwargs.

    Args:
        kwargs: Normalized generation model kwargs mapping. May be ``None``.

    Returns:
        dict: Compact dictionary with only non-``None`` entries.
    """

    if not kwargs:
        return {}
    return {key: value for key, value in kwargs.items() if value is not None}


def _flatten_grouped_multimodal_model_value(key: str, value: tp.Any) -> tp.Any:
    """Flatten grouped multimodal leaves before the actual model call."""

    if key not in GROUPED_MULTIMODAL_MODEL_INPUT_KEYS or not hasattr(value, "shape"):
        return value

    if key == "pixel_values" and value.ndim >= 5:
        return jnp.reshape(value, (-1, *value.shape[2:]))
    if key == "pixel_values_videos" and value.ndim >= 6:
        return jnp.reshape(value, (-1, *value.shape[2:]))
    if key in {"image_grid_thw", "video_grid_thw"} and value.ndim >= 3 and value.shape[-1] == 3:
        return jnp.reshape(value, (-1, 3))
    if key in {"image_grid_hws", "image_sizes"} and value.ndim >= 3 and value.shape[-1] == 2:
        return jnp.reshape(value, (-1, 2))
    return value


def _extend_prompt_scoring_value_to_sequence_length(
    key: str,
    value: tp.Any,
    *,
    prompt_length: int | None,
    target_sequence_length: int | None,
) -> tp.Any:
    """Extend prompt-only token-type style tensors across generated text tokens."""

    if key not in PROMPT_SCORING_EXTENSION_KEYS or not hasattr(value, "shape"):
        return value
    if prompt_length is None or target_sequence_length is None or value.ndim == 0:
        return value
    current_length = value.shape[-1]
    if current_length != prompt_length or target_sequence_length <= current_length:
        return value

    pad_width = [(0, 0)] * value.ndim
    pad_width[-1] = (0, target_sequence_length - current_length)
    pad_value = 0.0 if jnp.issubdtype(jnp.asarray(value).dtype, jnp.floating) else 0
    return jnp.pad(jnp.asarray(value), pad_width, mode="constant", constant_values=pad_value)


def prepare_generation_model_kwargs_for_call(
    kwargs: collections.abc.Mapping[str, tp.Any] | None,
    *,
    target_sequence_length: int | None = None,
    prompt_length: int | None = None,
    flatten_grouped_multimodal: bool = True,
) -> dict[str, tp.Any]:
    """Prepare generation kwargs for a model call without losing prompt grouping upstream.

    Args:
        kwargs: Compact generation model kwargs mapping. May be ``None``.
        target_sequence_length: Target sequence length for extending prompt-only
            tensors. Defaults to ``None``.
        prompt_length: Length of the prompt portion. Defaults to ``None``.
        flatten_grouped_multimodal: Whether grouped multimodal leaves should be
            flattened to model-call layout immediately. Generation should keep
            prompt grouping until after any batch expansion.

    Returns:
        dict: Prepared dictionary ready for a model forward call.
    """

    prepared: dict[str, tp.Any] = {}
    for key, value in compact_generation_model_kwargs(kwargs).items():
        if flatten_grouped_multimodal:
            value = _flatten_grouped_multimodal_model_value(key, value)
        value = _extend_prompt_scoring_value_to_sequence_length(
            key,
            value,
            prompt_length=prompt_length,
            target_sequence_length=target_sequence_length,
        )
        prepared[key] = value
    return prepared


def strip_prompt_only_scoring_model_kwargs(
    kwargs: collections.abc.Mapping[str, tp.Any] | None,
) -> dict[str, tp.Any]:
    """Drop prompt-only sequence-control kwargs before full-sequence scoring.

    Args:
        kwargs: Generation model kwargs mapping. May be ``None``.

    Returns:
        dict: Kwargs with prompt-only scoring keys removed.
    """

    compact_kwargs = compact_generation_model_kwargs(kwargs)
    if (
        compact_kwargs.get("deepstack_visual_embeds", None) is not None
        and compact_kwargs.get("visual_pos_masks", None) is not None
    ):
        excluded_keys = PROMPT_ONLY_SCORING_MODEL_INPUT_KEYS - {"visual_pos_masks"}
    else:
        excluded_keys = PROMPT_ONLY_SCORING_MODEL_INPUT_KEYS
    return {key: value for key, value in compact_kwargs.items() if key not in excluded_keys}


def extract_generation_model_kwargs(
    batch: collections.abc.Mapping[str, tp.Any] | None,
    *,
    model_callable: tp.Callable[..., tp.Any] | None = None,
) -> dict[str, tp.Any]:
    """Extract generation-related model inputs from a larger batch mapping.

    Args:
        batch: Batch mapping potentially containing generation model inputs.
            May be ``None``.
        model_callable: Optional model forward callable used to filter keys
            against its signature.

    Returns:
        dict: Compact dictionary of generation-related model inputs.
    """

    return compact_generation_model_kwargs(
        normalize_generation_model_kwargs(batch, model_callable=model_callable),
    )


def validate_prompt_aligned_generation_model_kwargs(
    kwargs: collections.abc.Mapping[str, tp.Any] | None,
    *,
    prompt_batch_size: int | None,
) -> None:
    """Validate that generation kwargs preserve prompt boundaries for GRPO-style scoring.

    Args:
        kwargs: Generation model kwargs to validate. May be ``None``.
        prompt_batch_size: Expected prompt batch size that kwargs should align with.

    Raises:
        ValueError: If any non-shared kwarg doesn't expose the prompt batch size
            on any axis.
    """

    compact_kwargs = compact_generation_model_kwargs(kwargs)
    if prompt_batch_size is None or prompt_batch_size <= 0:
        return

    for key, value in compact_kwargs.items():
        if key in SHARED_GENERATION_MODEL_INPUT_KEYS:
            continue
        if infer_prompt_batch_axis(value, prompt_batch_size, key=key) is not None:
            continue

        shape = getattr(value, "shape", None)
        raise ValueError(
            "GRPO requires prompt-aligned generation kwargs for scoring. "
            f"Got `{key}` with shape {shape!r}, which does not expose the prompt batch size "
            f"{prompt_batch_size} on any axis. Raw ragged multimodal inputs are not supported "
            "here; use prompt-batch-aligned embeddings/features or a single aligned item per prompt."
        )


def infer_prompt_batch_axis(
    value: tp.Any,
    prompt_batch_size: int | None,
    *,
    key: str | None = None,
) -> int | None:
    """Infer which axis of ``value`` is aligned with the prompt batch.

    Args:
        value: Array, list, or tuple whose batch axis is to be inferred.
        prompt_batch_size: The expected prompt batch size.
        key: Optional key name for special-case handling (e.g. ``"position_ids"``).

    Returns:
        int | None: The axis index aligned with the prompt batch, or ``None``
            if no axis matches.
    """

    if prompt_batch_size is None or prompt_batch_size <= 0:
        return None

    if isinstance(value, (list, tuple)):
        return 0 if len(value) == prompt_batch_size else None

    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        return None

    if key == "position_ids" and len(shape) >= 3 and shape[0] == 3 and shape[1] == prompt_batch_size:
        return 1
    if shape[0] == prompt_batch_size:
        return 0
    return None


def repeat_prompt_aligned_model_value(
    value: tp.Any,
    repeat_factor: int,
    *,
    prompt_batch_size: int | None = None,
    key: str | None = None,
) -> tp.Any:
    """Repeat a prompt-aligned model input along its actual batch axis.

    Args:
        value: Array, list, or tuple to repeat.
        repeat_factor: Number of times to repeat each element.
        prompt_batch_size: Expected prompt batch size for axis inference.
        key: Optional key name for special-case axis inference.

    Returns:
        The repeated value along its batch axis, or the original value if
        ``repeat_factor <= 1`` or no batch axis is found.
    """

    if repeat_factor <= 1:
        return value

    batch_axis = infer_prompt_batch_axis(value, prompt_batch_size, key=key)
    if batch_axis is None:
        return value

    if isinstance(value, list):
        return [item for item in value for _ in range(repeat_factor)]
    if isinstance(value, tuple):
        return tuple(item for item in value for _ in range(repeat_factor))

    return jnp.repeat(jnp.asarray(value), repeat_factor, axis=batch_axis)


def slice_prompt_aligned_model_value(
    value: tp.Any,
    start: int,
    end: int,
    *,
    prompt_batch_size: int | None = None,
    key: str | None = None,
) -> tp.Any:
    """Slice a prompt-aligned model input along its actual batch axis.

    Args:
        value: Array, list, or tuple to slice.
        start: Start index of the slice.
        end: End index of the slice.
        prompt_batch_size: Expected prompt batch size for axis inference.
        key: Optional key name for special-case axis inference.

    Returns:
        The sliced value along its batch axis, or the original value if
        no batch axis is found.
    """

    batch_axis = infer_prompt_batch_axis(value, prompt_batch_size, key=key)
    if batch_axis is None:
        return value

    if isinstance(value, list):
        return value[start:end]
    if isinstance(value, tuple):
        return value[start:end]

    index = [slice(None)] * value.ndim
    index[batch_axis] = slice(start, end)
    return value[tuple(index)]


def slice_prompt_aligned_model_kwargs(
    kwargs: collections.abc.Mapping[str, tp.Any],
    start: int,
    end: int,
    *,
    prompt_batch_size: int | None = None,
) -> dict[str, tp.Any]:
    """Slice prompt-aligned model kwargs while preserving shared leaves.

    Args:
        kwargs: Mapping of model kwargs to slice.
        start: Start index of the slice.
        end: End index of the slice.
        prompt_batch_size: Expected prompt batch size for axis inference.

    Returns:
        dict: Sliced kwargs dictionary.
    """

    sliced: dict[str, tp.Any] = {}
    for key, value in kwargs.items():
        if value is None:
            sliced[key] = None
            continue
        sliced[key] = slice_prompt_aligned_model_value(
            value,
            start,
            end,
            prompt_batch_size=prompt_batch_size,
            key=key,
        )
    return sliced


def repeat_prompt_aligned_model_kwargs(
    kwargs: collections.abc.Mapping[str, tp.Any] | None,
    repeat_factor: int,
    *,
    prompt_batch_size: int | None = None,
) -> dict[str, tp.Any]:
    """Repeat prompt-aligned model kwargs to match completion-aligned batches.

    Args:
        kwargs: Mapping of model kwargs to repeat. May be ``None``.
        repeat_factor: Number of times to repeat each element along its batch axis.
        prompt_batch_size: Expected prompt batch size for axis inference.

    Returns:
        dict: Repeated kwargs dictionary.
    """

    compact_kwargs = compact_generation_model_kwargs(kwargs)
    if repeat_factor <= 1 or not compact_kwargs:
        return dict(compact_kwargs)

    repeated: dict[str, tp.Any] = {}
    for key, value in compact_kwargs.items():
        repeated[key] = repeat_prompt_aligned_model_value(
            value,
            repeat_factor,
            prompt_batch_size=prompt_batch_size,
            key=key,
        )
    return repeated


def _ste(x: jax.Array, q: jax.Array) -> jax.Array:
    """Straight-through estimator: ``q`` on the forward pass, identity on the backward pass.

    Args:
        x: The original (full-precision) tensor.
        q: A quantized approximation of ``x``.

    Returns:
        ``x + stop_gradient(q - x)``, equal to ``q`` numerically while
        passing gradients through unchanged.
    """
    q = q.astype(x.dtype)
    return x + lax.stop_gradient(q - x)


def make_default_tensor_straight_through(
    quantization_mode: QuantizationMode,
    quantization_group_size: int | None = None,
    quantization_bits: int | None = None,
    *,
    quantization_block: int | None = None,
) -> tp.Callable[[jax.Array], jax.Array]:
    """Create a per-tensor STE quantization function.

    Forward path uses a quantize->dequantize simulation, while gradients flow as
    if the transform is identity (STE).

    Notes:
        - `quantization_group_size` controls group-wise quantization where relevant.
        - `quantization_bits` controls bit-width for configurable formats (for example `affine`).
    """
    if quantization_block is not None:
        warnings.warn(
            "`quantization_block` is deprecated; use `quantization_group_size` instead.",
            FutureWarning,
            stacklevel=2,
        )
        if quantization_group_size is None:
            quantization_group_size = quantization_block
        elif quantization_group_size != quantization_block:
            warnings.warn(
                f"Both `quantization_group_size` ({quantization_group_size}) and "
                f"`quantization_block` ({quantization_block}) are set; ignoring `quantization_block`.",
                FutureWarning,
                stacklevel=2,
            )

    if quantization_bits is not None:
        quantization_bits = int(quantization_bits)
        if quantization_bits <= 0:
            raise ValueError(f"`quantization_bits` must be > 0 when specified, got {quantization_bits}.")
        if quantization_mode == "affine" and quantization_bits not in AFFINE_SUPPORTED_BITS:
            bits_values = ", ".join(str(v) for v in sorted(AFFINE_SUPPORTED_BITS))
            raise ValueError(
                f"`quantization_bits` for `affine` must be one of {{{bits_values}}}, got {quantization_bits}."
            )
        required_bits = FIXED_QUANTIZATION_BITS_BY_MODE.get(quantization_mode, None)
        if required_bits is not None and quantization_bits != required_bits:
            raise ValueError(
                f"`quantization_bits` for `{quantization_mode}` must be {required_bits}, got {quantization_bits}."
            )

    from ejkernel.quantization import dequantize as ej_dequantize  # pyright: ignore[reportMissingTypeStubs]
    from ejkernel.quantization import quantize as ej_quantize  # pyright: ignore[reportMissingTypeStubs]

    from easydel.layers.quantization import QuantizationConfig
    from easydel.layers.quantization._configs import resolve_ejkernel_quant_params

    quantization_config = QuantizationConfig(
        dtype=quantization_mode,
        group_size=quantization_group_size,
        bits=quantization_bits,
    )
    mode, group_size, bits, needs_biases = resolve_ejkernel_quant_params(quantization_config)

    def _quantize_dequantize(y: jax.Array) -> jax.Array:
        """Simulate the quantize/dequantize round-trip for a single tensor leaf.

        Handles 0-d and 1-d edge cases, pads the last dim to a multiple of
        ``group_size``, and dispatches to the appropriate ejkernel quantizer
        (with or without zero-points / biases).

        Args:
            y: Float tensor to round-trip through the quantization scheme.

        Returns:
            A tensor of the same shape and dtype as ``y`` whose values are
            the quantize-then-dequantize image of ``y``.
        """
        input_dtype = y.dtype
        if y.ndim == 0:
            # Scalar leaves can appear in graphstate pytrees; keep them unchanged.
            return y.astype(input_dtype)
        was_vector = y.ndim == 1
        if was_vector:
            # ejkernel quantize expects rank >= 2.
            y = y[None, :]
        original_last_dim = y.shape[-1]
        if original_last_dim % group_size != 0:
            pad_amount = group_size - (original_last_dim % group_size)
            pad_width = [(0, 0)] * (y.ndim - 1) + [(0, pad_amount)]
            y = jnp.pad(y, pad_width, mode="constant", constant_values=0)

        if needs_biases:
            wq, scales, biases = ej_quantize(y, group_size=group_size, bits=bits, mode=mode, axis="col")
        else:
            wq, scales = ej_quantize(y, group_size=group_size, bits=bits, mode=mode, axis="col")
            biases = None
        dequantized = ej_dequantize(
            wq,
            scales,
            biases,
            group_size=group_size,
            bits=bits,
            mode=mode,
            axis="col",
        )
        if dequantized.shape[-1] != original_last_dim:
            dequantized = dequantized[..., :original_last_dim]
        if was_vector:
            dequantized = jnp.squeeze(dequantized, axis=0)
        return dequantized.astype(input_dtype)

    def tensor_straight_through(x: jax.Array) -> jax.Array:
        """Apply STE quantization to a single tensor leaf.

        Non-floating tensors are returned unchanged so that integer
        bookkeeping leaves (e.g. step counters) are not perturbed.

        Args:
            x: Tensor leaf to quantize on the forward pass only.

        Returns:
            The straight-through quantized tensor.
        """
        if not jnp.issubdtype(x.dtype, jnp.floating):
            return x
        return _ste(x, _quantize_dequantize(x))

    return tensor_straight_through


def resolve_straight_through_emulator(
    *,
    quantization_mode: QuantizationMode | None,
    quantization_group_size: int | None = None,
    quantization_bits: int | None = None,
    tensor_straight_through: tp.Callable[[jax.Array], jax.Array] | None,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None,
    quantization_block: int | None = None,
) -> tp.Callable[[tp.Any], tp.Any] | None:
    """Resolve the graphstate-level straight-through emulator callable.

    Priority:
      1) `straight_through_emulator` (user-provided)
      2) `tensor_straight_through` mapped over graphstate
      3) default tensor STE built from (`quantization_mode`, `quantization_group_size`, `quantization_bits`) and
         mapped over graphstate
      4) None (disabled)
    """
    if quantization_block is not None:
        warnings.warn(
            "`quantization_block` is deprecated; use `quantization_group_size` instead.",
            FutureWarning,
            stacklevel=2,
        )
        if quantization_group_size is None:
            quantization_group_size = quantization_block
        elif quantization_group_size != quantization_block:
            warnings.warn(
                f"Both `quantization_group_size` ({quantization_group_size}) and "
                f"`quantization_block` ({quantization_block}) are set; ignoring `quantization_block`.",
                FutureWarning,
                stacklevel=2,
            )

    if straight_through_emulator is not None:
        return straight_through_emulator

    if tensor_straight_through is None and quantization_mode is None:
        return None

    if tensor_straight_through is None:
        tensor_straight_through = make_default_tensor_straight_through(
            quantization_mode,
            quantization_group_size=quantization_group_size,
            quantization_bits=quantization_bits,
        )

    def _default_emulator(graphstate: tp.Any) -> tp.Any:
        """Apply ``tensor_straight_through`` over every leaf of ``graphstate``.

        Args:
            graphstate: Pytree of tensor leaves (typically the model's
                graph-state).

        Returns:
            A pytree of identical shape with each float leaf passed through
            the per-tensor STE.
        """
        return tu.tree_map(tensor_straight_through, graphstate)

    return _default_emulator


def resolve_total_steps(
    *,
    forced_steps: int | None,
    total_data_len: int | None,
    batch_size: int,
    num_epochs: int,
    gradient_accumulation_steps: int,
    is_train: bool,
) -> int:
    """Resolve total train/eval steps from config and dataset length.

    Notes:
        - `forced_steps` is interpreted as *optimizer update* steps for training (i.e., after gradient accumulation).
        - When `forced_steps` is not provided, training steps are derived from dataset length and then divided by
          `gradient_accumulation_steps` to convert micro-batches into optimizer updates.
    """
    if forced_steps is not None:
        return int(forced_steps)

    if total_data_len is None:
        raise ValueError("`total_data_len` must be provided when `forced_steps` is None.")
    if batch_size <= 0:
        raise ValueError("`batch_size` must be > 0.")
    if num_epochs <= 0:
        return 0

    steps_per_epoch = (total_data_len + batch_size - 1) // batch_size
    steps = steps_per_epoch * num_epochs

    if is_train:
        if gradient_accumulation_steps <= 0:
            raise ValueError("`gradient_accumulation_steps` must be > 0.")
        steps //= gradient_accumulation_steps

    return int(steps)


def make_assertions_and_get_sizes(
    batch: dict,
    gradient_accumulation_steps: int,
    batch_partition_spec: PartitionSpec | None = None,
) -> tuple[int, int, PartitionSpec]:
    """
    Validates the input parameters and computes the batch size, minibatch size, and batch partition specification.
    Args:
        batch (tp.Dict): A dictionary containing the batch data. The batch size is inferred from the
            dominant leading dimension across array leaves.
        gradient_accumulation_steps (int): The number of gradient accumulation steps. Must be greater than 0.
        batch_partition_spec (tp.Optional[PartitionSpec], optional): The partition specification for the batch.
            Defaults to None.
    Returns:
        tp.Tuple[int, int, PartitionSpec]: A tuple containing:
            - batch_size (int): The size of the batch.
            - minibatch_size (int): The size of the minibatch.
            - batch_partition_spec (PartitionSpec): The partition specification for the batch.
    Raises:
            ValueError: If `gradient_accumulation_steps` is not greater than 0.
            ValueError: If the batch size is not divisible by the gradient accumulation steps.
    """

    if gradient_accumulation_steps <= 0:
        raise ValueError("`gradient_accumulation_steps` must be greater than 0.")

    batch_size = _infer_batch_size(batch)

    minibatch_size = batch_size // gradient_accumulation_steps
    if minibatch_size * gradient_accumulation_steps != batch_size:
        raise ValueError("Batch size must be divisible by gradient accumulation steps.")
    if batch_partition_spec is None:
        batch_partition_spec = PartitionSpec(("dp", "fsdp"), "sp")
    return batch_size, minibatch_size, batch_partition_spec


def _normalize_static_argnums(static_argnums: int | tp.Sequence[int] | None) -> tuple[int, ...]:
    """Coerce a static_argnums spec into a tuple of ints.

    Args:
        static_argnums: Either a single int, an iterable of ints, or
            ``None``.

    Returns:
        A (possibly empty) tuple of ints.
    """
    if static_argnums is None:
        return ()
    if isinstance(static_argnums, int):
        return (static_argnums,)
    return tuple(static_argnums)


def _normalize_static_argnames(static_argnames: str | tp.Iterable[str] | None) -> tuple[str, ...]:
    """Coerce a static_argnames spec into a tuple of names.

    Args:
        static_argnames: Either a single name, an iterable of names, or
            ``None``.

    Returns:
        A (possibly empty) tuple of names.
    """
    if static_argnames is None:
        return ()
    if isinstance(static_argnames, str):
        return (static_argnames,)
    return tuple(static_argnames)


def compile_trainer_step(
    fn: tp.Callable[..., tp.Any],
    *,
    mutable: tp.Any = (),
    mesh: MeshLike | None = None,
    schedule: MpMdSchedulers | None = None,
    arguments: tp.Any | None = None,
    in_shardings: tp.Any = _UNSPECIFIED,
    out_shardings: tp.Any = _UNSPECIFIED,
    static_argnums: int | tp.Sequence[int] | None = None,
    static_argnames: str | tp.Iterable[str] | None = None,
    donate_argnums: int | tp.Sequence[int] | None = None,
    donate_argnames: str | tp.Iterable[str] | None = None,
    batch_argnums: int | tp.Sequence[int] | None = None,
    keep_unused: bool = False,
    **jit_kwargs,
) -> tp.Callable[..., tp.Any]:
    """Compile a trainer step with the SpectraX MPMD path when the mesh requires it.

    When ``schedule`` is None, falls back to ``arguments.mpmd_scheduler`` if
    ``arguments`` is supplied -- so trainers can opt in to 1F1B / GPipe
    microbatching by setting ``TrainingArguments.mpmd_scheduler`` once,
    without touching every trainer's call site. Trainer steps that register a
    :class:`ScheduledLossAdapter` are run through the shared scheduled-VJP path;
    unregistered full trainer steps keep the regular marker-based JIT path.
    """

    if schedule is None and arguments is not None:
        schedule = getattr(arguments, "mpmd_scheduler", None)
    scheduled_adapter = get_scheduled_loss_adapter(fn) if schedule is not None else None
    if scheduled_adapter is not None:
        return _compile_scheduled_training_step(
            step_fn=fn,
            mesh=mesh,
            schedule=schedule,
            batch_argnums=(1,) if batch_argnums is None else batch_argnums,
            static_argnums=static_argnums,
            adapter=scheduled_adapter,
        )
    # SpectraX's schedule= path is a scalar-loss custom-VJP runtime. Whole
    # EasyDeL trainer steps usually return (state, metrics) or metrics, and
    # auxiliary forwards return logits/log-probs. Keep all unregistered
    # callables on the regular marker-based MPMD path so custom trainers keep
    # their full metrics and is_training behavior.

    static_nums = _normalize_static_argnums(static_argnums)
    kwargs = {
        "mutable": mutable,
        "static_argnums": static_argnums,
        "static_argnames": static_argnames,
        "donate_argnums": donate_argnums,
        "donate_argnames": donate_argnames,
        "keep_unused": keep_unused,
        **jit_kwargs,
    }
    if mesh is not None:
        kwargs["mesh"] = mesh
    if in_shardings is not _UNSPECIFIED:
        kwargs["in_shardings"] = in_shardings
    if out_shardings is not _UNSPECIFIED:
        kwargs["out_shardings"] = out_shardings
    compiled = spx.jit(fn, **kwargs)
    compiled.static_argnums_ = static_nums
    return compiled


def _slice_batch_for_scheduled_step(batch: dict, batch_size: int, start_index: int, minibatch_size: int) -> dict:
    """Slice leading-batch leaves while passing shared leaves through."""

    def _slice_leaf(arr):
        """Slice a single leaf along axis 0 when it carries the full batch dim.

        Args:
            arr: A pytree leaf (typically a JAX array).

        Returns:
            The dynamically sliced minibatch view, or ``arr`` unchanged
            when it does not carry the leading batch dimension.
        """
        if not hasattr(arr, "shape") or arr.ndim == 0:
            return arr
        if arr.shape[0] == batch_size:
            return lax.dynamic_slice_in_dim(arr, start_index, minibatch_size, axis=0)
        return arr

    return jax.tree_util.tree_map(_slice_leaf, batch)


def _scheduled_step_key(fn: tp.Callable[..., tp.Any]) -> tuple[str, str]:
    """Return a stable ``(module, qualname)`` key for an unwrapped step function.

    Unwraps :class:`functools.partial` so registrations made on the
    original function are still discoverable through partial wrappers.

    Args:
        fn: The trainer step callable.

    Returns:
        A tuple of strings suitable for use as a dictionary key.
    """
    while isinstance(fn, functools.partial):
        fn = fn.func
    return getattr(fn, "__module__", ""), getattr(fn, "__name__", "")


def _scheduled_step_name(fn: tp.Callable[..., tp.Any]) -> str:
    """Return a human-readable name for a (possibly partial) step function.

    Args:
        fn: The trainer step callable.

    Returns:
        The function name when available, otherwise its module name or
        the class name of the wrapper.
    """
    module, name = _scheduled_step_key(fn)
    return name or module or type(fn).__name__


def _scheduled_terminal_stage_rank(module: tp.Any, schedule: tp.Any) -> int | None:
    """Physical MPMD rank that hosts the terminal (loss) pipeline stage, or ``None``.

    Used by scheduled-loss closures so model-side ``spx.with_sharding_constraint``
    calls (e.g. inside :func:`module.make_lm_head_fn` or chunked-CE projectors)
    can name the right stage submesh. A bare constraint there would resolve
    per-process on a multi-stage mesh and miscompile, since the loss runs
    outside any ``spx.assign_stage`` context.

    Returns ``None`` when there is no multi-stage pipeline (single-rank,
    no mesh, or any inspection failure) -- callers should treat that as
    "no constraint" (safe no-op) rather than a miscompile.
    """
    try:
        mesh = getattr(module, "mesh", None)
        n = getattr(mesh, "mpmd_dim", None)
        if n is None and hasattr(module, "_pipeline_physical_stage_count"):
            n = int(module._pipeline_physical_stage_count())
        if n is None or int(n) <= 1:
            return None
        terminal_loc = schedule.terminal_loc(int(n))
        if isinstance(terminal_loc, (tuple, list)):
            return int(terminal_loc[0])
        return int(terminal_loc)
    except Exception:
        return None


def _mpmd_host_replicate_scalar(value: tp.Any) -> tp.Any:
    """Make a scheduled-MPMD scalar (e.g. the step loss) host-fetchable on every controller.

    The scheduled VJP computes its scalar loss on the *terminal* pipeline stage's device
    submesh. Under multi-controller (multi-host) execution every controller process that
    does not own a device of that submesh holds a ``jax.Array`` with **zero** addressable
    shards, so ``jax.device_get(loss)`` (used for the ``break_on_nan`` guard and by the
    trainer for metrics/logging) raises ``Fetching value for jax.Array that spans
    non-addressable (non process local) devices``.

    This broadcasts the value from the single lowest-indexed owning process to **all**
    processes through an all-processes psum (``broadcast_one_to_all``), so host-side
    bookkeeping works. ``NaN``/``Inf`` are preserved (``nan + 0 + ... == nan``), so the
    NaN guard still fires correctly. Single-process runs are returned unchanged.

    Note: we deliberately do **not** short-circuit on ``value.is_fully_addressable`` --
    that predicate is process-local, so doing so could make some controllers skip the
    collective while others enter it (deadlock). When ``process_count > 1`` every
    controller must take the same branch.
    """
    try:
        proc_count = int(jax.process_count())
    except Exception:
        return value
    if proc_count <= 1:
        return value
    sharding = getattr(value, "sharding", None)
    if sharding is None:
        return value
    try:
        owner_procs = sorted({int(d.process_index) for d in sharding.device_set})
    except Exception:
        return value
    if not owner_procs:
        return value
    from jax.experimental import multihost_utils

    src_proc = owner_procs[0]
    is_source = int(jax.process_index()) == src_proc
    if is_source:
        local = np.asarray(jax.device_get(value)).astype(np.float32).reshape(())
    else:
        local = np.zeros((), dtype=np.float32)
    gathered = multihost_utils.broadcast_one_to_all(local, is_source=is_source)
    return jnp.asarray(np.asarray(gathered).reshape(()))


def _apply_stage_local_gradients(
    *,
    state: EasyDeLState,
    gradients: tp.Any,
    loss: jax.Array,
    loss_config: LossConfig | None,
    learning_rate_fn: tp.Any,
) -> tuple[EasyDeLState, LossMetrics]:
    """Apply stage-local gradients via the optimizer's PP-aware update path.

    Used by the scheduled training-step path so each pipeline stage updates
    only its local shard of parameters and optimizer state.  Honors a
    ``break_on_nan`` LossConfig by returning the unchanged state when the
    loss is ``NaN``.

    Args:
        state: Current model/optimizer state.
        gradients: Stage-local gradient pytree.
        loss: Scalar loss value used for metrics and NaN detection.
        loss_config: Optional :class:`LossConfig`; ``break_on_nan`` is
            consulted.
        learning_rate_fn: Schedule function for the optimizer.

    Returns:
        ``(new_state, metrics)`` after applying the optimizer update; or
        ``(state, metrics)`` unchanged on a NaN loss when ``break_on_nan``
        is set.

    Raises:
        RuntimeError: If the state's optimizer is missing or does not
            implement :meth:`apply_gradients_stage_local`.
    """
    metrics = update_metrics(
        metrics=LossMetrics(loss=loss),
        learning_rate_fn=learning_rate_fn,
        step=state.step,
        gradients=None,
    )
    if loss_config is not None and bool(getattr(loss_config, "break_on_nan", False)):
        if bool(jax.device_get(jnp.isnan(loss))):
            return state, metrics

    if state.tx is None:
        raise RuntimeError("mpmd_scheduler requires an initialized optimizer transformation.")
    if state.opt_state is None:
        raise RuntimeError("mpmd_scheduler requires initialized optimizer state.")

    apply_stage_local = getattr(state.tx, "apply_gradients_stage_local", None)
    if not callable(apply_stage_local):
        raise RuntimeError(
            "mpmd_scheduler produced stage-local gradients, but the optimizer does not expose "
            "`apply_gradients_stage_local`. Use an eFormer optimizer with PP stage-local support."
        )

    try:
        graphstate, opt_state = apply_stage_local(
            params=state.graphstate,
            grads=gradients,
            opt_state=state.opt_state,
            learning_rate_fn=learning_rate_fn,
            delete_grads=True,
        )
    except NotImplementedError as exc:
        raise RuntimeError(f"Optimizer does not support PP stage-local updates: {exc}") from exc

    new_state = state.replace(step=state.step + 1, graphstate=graphstate, opt_state=opt_state)
    return new_state, metrics


def _run_scheduled_value_and_grad(
    *,
    value_and_grad: _ScheduledValueAndGradFn,
    graphstate: tp.Any,
    batch: dict,
    batch_size: int,
    minibatch_size: int,
) -> tuple[jax.Array, tp.Any]:
    """Run the scheduled value-and-grad with optional gradient accumulation.

    When ``batch_size > minibatch_size``, the input batch is split into
    equal-sized minibatches and the gradients are averaged.

    Args:
        value_and_grad: The compiled scheduled value-and-grad callable.
        graphstate: The state pytree to differentiate against.
        batch: The full minibatch dictionary.
        batch_size: Total batch size (leading dimension of ``batch``).
        minibatch_size: Size of each accumulation step.

    Returns:
        ``(loss, gradients)`` aggregated across all accumulation steps.
    """
    num_accum_steps = batch_size // minibatch_size
    if num_accum_steps == 1:
        return value_and_grad(graphstate, batch)

    loss_acc = None
    grad_acc = None
    for accum_idx in range(num_accum_steps):
        minibatch = _slice_batch_for_scheduled_step(
            batch,
            batch_size,
            accum_idx * minibatch_size,
            minibatch_size,
        )
        loss_i, gradients_i = value_and_grad(graphstate, minibatch)
        loss_acc = loss_i if loss_acc is None else loss_acc + loss_i
        grad_acc = gradients_i if grad_acc is None else jax.tree_util.tree_map(jnp.add, grad_acc, gradients_i)

    inv_steps = jnp.asarray(1.0 / num_accum_steps, dtype=jnp.float32)
    if loss_acc is None or grad_acc is None:
        raise ValueError("Gradient accumulation produced no minibatches.")
    return loss_acc * inv_steps, jax.tree_util.tree_map(lambda x: x * inv_steps, grad_acc)


def _compile_scheduled_training_step(
    *,
    step_fn: tp.Callable[..., tp.Any],
    mesh: MeshLike | None,
    schedule: tp.Any,
    batch_argnums: int | tp.Sequence[int] | None,
    static_argnums: int | tp.Sequence[int] | None,
    adapter: ScheduledLossAdapter,
) -> tp.Callable[..., tp.Any]:
    """Build the scheduled-VJP version of a registered trainer step.

    Wraps the trainer-supplied ``step_fn`` with a SpectraX-scheduled
    value-and-grad path: gradient accumulation is run via
    :func:`_run_scheduled_value_and_grad` and the resulting stage-local
    gradients are applied through :func:`_apply_stage_local_gradients`.

    Args:
        step_fn: The original trainer step function.
        mesh: Device mesh used for the scheduled compilation.
        schedule: The MPMD pipeline schedule (e.g. 1F1B, GPipe).
        batch_argnums: Position(s) of batch-typed arguments inside the
            trainer step signature.
        static_argnums: Position(s) of static (compile-time constant)
            arguments.
        adapter: The :class:`ScheduledLossAdapter` describing how to
            extract the scalar loss from ``step_fn``.

    Returns:
        A callable with the same ``(state, batch, ...)`` signature as
        ``step_fn`` that runs through the scheduled VJP path.

    Raises:
        ValueError: If ``mesh`` is ``None``.
    """
    if mesh is None:
        raise ValueError("mpmd_scheduler requires compile_trainer_step(..., mesh=...).")

    step_signature = inspect.signature(step_fn)
    scheduled_vag = _ScheduledValueAndGradCompiler(
        mesh=mesh,
        schedule=schedule,
        batch_argnums=batch_argnums,
        adapter=adapter,
    )
    _ScheduledValueAndGradAndApplyCompiler(
        mesh=mesh,
        schedule=schedule,
        batch_argnums=batch_argnums,
        adapter=adapter,
    )

    def scheduled_training_step(
        state: EasyDeLState,
        batch: collections.abc.Mapping[str, jax.Array],
        *step_args: tp.Any,
        **step_kwargs: tp.Any,
    ) -> tuple[EasyDeLState, LossMetrics]:
        """Run one scheduled-VJP training step with optional gradient accumulation.

        Args:
            state: Current model/optimizer state.
            batch: Minibatch dictionary of sharded JAX arrays.
            *step_args: Forwarded positional arguments matching
                ``step_fn``'s signature.
            **step_kwargs: Forwarded keyword arguments.

        Returns:
            ``(new_state, metrics)`` after applying the optimizer update.
        """
        batch = dict(batch)
        bound = step_signature.bind(state, batch, *step_args, **step_kwargs)
        bound.apply_defaults()
        bound_arguments = dict(bound.arguments)
        loss_config = bound_arguments.get("loss_config")
        learning_rate_fn = bound_arguments.get("learning_rate_fn")
        gradient_accumulation_steps = bound_arguments.get("gradient_accumulation_steps", 1)
        batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
            batch=batch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            batch_partition_spec=bound_arguments.get("partition_spec"),
        )
        bound_arguments["batch"] = batch
        bound_arguments["partition_spec"] = partition_spec
        call = ScheduledStepCall(
            step_fn=step_fn,
            state=state,
            batch=batch,
            args=step_args,
            kwargs=step_kwargs,
            bound_arguments=bound_arguments,
            schedule=schedule,
        )
        if adapter.prepare_batch is not None:
            _t_prep = time.perf_counter()
            batch = dict(adapter.prepare_batch(call))
            _log_sched_section("prepare_batch (teacher/ref fwd)", _t_prep, batch)
            bound_arguments["batch"] = batch
            call = ScheduledStepCall(
                step_fn=step_fn,
                state=state,
                batch=batch,
                args=step_args,
                kwargs=step_kwargs,
                bound_arguments=bound_arguments,
                schedule=schedule,
            )

        value_and_grad = scheduled_vag.get(call)
        _t_vag = time.perf_counter()
        loss, gradients = _run_scheduled_value_and_grad(
            value_and_grad=value_and_grad,
            graphstate=state.graphstate,
            batch=batch,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
        )
        _log_sched_section("value_and_grad (student fwd+bwd)", _t_vag, (loss, gradients))
        loss = _mpmd_host_replicate_scalar(loss)
        _t_opt = time.perf_counter()
        out = _apply_stage_local_gradients(
            state=state,
            gradients=gradients,
            loss=loss,
            loss_config=loss_config,
            learning_rate_fn=learning_rate_fn,
        )
        _log_sched_section("apply_stage_local_gradients (opt)", _t_opt, out)
        return out

    scheduled_training_step.__name__ = f"{type(schedule).__name__}_{adapter.name}_{_scheduled_step_name(step_fn)}"
    scheduled_training_step.static_argnums_ = _normalize_static_argnums(static_argnums)
    return scheduled_training_step


def compile_trainer_auxiliary(
    fn: tp.Callable[..., tp.Any],
    *,
    mesh: MeshLike | None = None,
    arguments: tp.Any | None = None,
    in_shardings: tp.Any = _UNSPECIFIED,
    out_shardings: tp.Any = _UNSPECIFIED,
    **jit_kwargs,
) -> tp.Callable[..., tp.Any]:
    """Compile a nested trainer helper through the shared SpectraX jit surface.

    Convenience wrapper around :func:`compile_trainer_step` for helpers
    that are not primary training/evaluation step functions (e.g.
    teacher/reference forwards, sample generators).

    Args:
        fn: The auxiliary helper callable.
        mesh: Optional device mesh override forwarded to ``spx.jit``.
        arguments: Optional :class:`TrainingArguments` instance used to
            inherit MPMD schedule configuration.
        in_shardings: Optional input sharding override.
        out_shardings: Optional output sharding override.
        **jit_kwargs: Additional keyword arguments forwarded verbatim
            to ``spx.jit``.

    Returns:
        A jit-compiled callable wrapping ``fn``.
    """

    return compile_trainer_step(
        fn,
        mesh=mesh,
        arguments=arguments,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        **jit_kwargs,
    )


def _infer_batch_size(batch: tp.Any) -> int:
    """Infer batch size from the most common leading dimension in the batch pytree.

    Walks the leaves of ``batch``, collects each leaf's leading axis
    size, and returns the mode. Robust to extra leaves whose first
    dimension does not represent the batch (e.g. scalar metadata or
    multimodal tensors with their own leading axis).

    Args:
        batch: Batch pytree (typically a dict of JAX arrays).

    Returns:
        The most frequent leading-axis size across array leaves.

    Raises:
        ValueError: If no array-typed leaf exposes a leading dimension.
    """

    leading_dims = [
        int(leaf.shape[0])
        for leaf in tu.tree_leaves(batch)
        if hasattr(leaf, "shape") and len(getattr(leaf, "shape", ())) >= 1
    ]
    if not leading_dims:
        raise ValueError(
            "Unable to infer batch size from `batch`; expected at least one array leaf with a leading batch dimension."
        )
    return collections.Counter(leading_dims).most_common(1)[0][0]


def update_metrics(
    metrics: LossMetrics,
    learning_rate_fn: tp.Callable,
    step: int | jax.Array,
    gradients: jax.Array | None,
) -> LossMetrics:
    """
    Updates the given metrics with the current learning rate and gradient norms.

    Args:
            metrics (LossMetrics): An instance of LossMetrics to be updated.
            learning_rate_fn (tp.Callable): A callable that returns the learning rate given the current step.
            step (int | jax.Array): The current training step.
            gradients (Optional(jax.Array)): The gradients to compute norms from.

    Returns:
            LossMetrics: The updated metrics with learning rate and gradient norms.
    """
    if learning_rate_fn is not None:
        metrics.learning_rate = learning_rate_fn(step)
    if gradients is not None:
        grad_norms = tu.tree_map(jnp.linalg.norm, gradients)
        metrics.max_grad_norm = tu.tree_reduce(jnp.maximum, grad_norms)
        grad_size = tu.tree_reduce(jnp.add, tu.tree_map(jnp.size, grad_norms))
        grad_sum = tu.tree_reduce(jnp.add, tu.tree_map(jnp.sum, grad_norms))
        metrics.mean_grad_norm = grad_sum / grad_size
        metrics.grad_norms = grad_norms
    return metrics


def update_state_respectfully(
    state: EasyDeLState,
    gradients: jax.Array,
    loss_config: LossConfig | None,
    metrics: LossMetrics,
) -> EasyDeLState:
    """
    Updates the state of the model respectfully based on the provided gradients, loss configuration, and metrics.

    Args:
            state (EasyDeLState): The current state of the model.
            gradients (jax.Array): The gradients to be applied to the model's parameters.
            loss_config (LossConfig): Configuration for the loss, including conditions for breaking on NaN values.
            metrics (LossMetrics): Metrics containing the loss value.

    Returns:
            EasyDeLState: The updated state of the model.
    """
    if FAST_COMPILE:
        return state.apply_gradients(grads=gradients)
    else:

        def update_fn(args):
            """Apply ``gradients`` to ``state`` via ``apply_gradients``.

            Args:
                args: ``(state, gradients)`` tuple.

            Returns:
                The updated state.
            """
            state, gradients = args
            return state.apply_gradients(grads=gradients)

        def skip_fn(args):
            """Return ``state`` unchanged (used when the gradient step is skipped).

            Args:
                args: ``(state, gradients)`` tuple; gradients are ignored.

            Returns:
                The original state.
            """
            state, _ = args
            return state

        should_update = True
        if loss_config is not None:
            should_update = lax.cond(
                loss_config.break_on_nan,
                lambda x: lax.cond(
                    jnp.isnan(metrics.loss),
                    lambda _: False,
                    lambda _: True,
                    None,
                ),
                lambda x: True,
                None,
            )
        state = lax.cond(should_update, update_fn, skip_fn, (state, gradients))
        return state


def minibatch_call(
    state: EasyDeLState,
    batch: dict,
    minibatch_size: int,
    grad_fn: tp.Callable[[jax.Array, dict], tuple[jax.Array, LossMetrics]],
) -> tuple[jax.Array, LossMetrics]:
    """
    Processes batch in smaller chunks for gradient accumulation using jax.lax.scan.

    Rather than reshaping the whole batch into
    ``(num_accum_steps, minibatch_size, ...)``, this slices minibatches from the
    original batch inside the scan body. That is friendlier to sharded arrays
    coming from model forwards (for example cached teacher hidden states in
    distillation), where introducing a new leading accumulation axis can confuse
    downstream partitioned computations.
    """
    batch_size = _infer_batch_size(batch)
    if minibatch_size <= 0:
        raise ValueError(f"`minibatch_size` must be > 0, got {minibatch_size}.")

    num_accum_steps = batch_size // minibatch_size
    if num_accum_steps * minibatch_size != batch_size:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by minibatch_size "
            f"({minibatch_size}) for gradient accumulation."
        )
    if num_accum_steps > 1:

        def slice_minibatch(tree, start_index):
            """Extract one minibatch while leaving shared/global leaves untouched."""

            def _slice_leaf(arr):
                """Slice a single leaf along axis 0 if it carries the full batch dim.

                Args:
                    arr: Pytree leaf (typically a JAX array).

                Returns:
                    The minibatch slice (when ``arr`` has the full batch
                    leading dim) or ``arr`` unchanged otherwise.
                """
                if not hasattr(arr, "shape") or arr.ndim == 0:
                    return arr
                if arr.shape[0] == batch_size:
                    return lax.dynamic_slice_in_dim(arr, start_index, minibatch_size, axis=0)
                return arr

            return jax.tree_util.tree_map(_slice_leaf, tree)

        shape_minibatch = slice_minibatch(batch, 0)

        (_, metrics_shape), grads_shape = jax.eval_shape(
            grad_fn,
            state.graphstate,
            shape_minibatch,
        )

        init_acc = {
            "grads": jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shape),
            "metrics": jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), metrics_shape),
        }

        def accumulate_gradients(acc, start_index):
            """Accumulate gradients and metrics for each minibatch."""
            minibatch = slice_minibatch(batch, start_index)
            (_, step_aux), step_grads = grad_fn(state.graphstate, minibatch)
            new_acc = {
                "grads": jax.tree_util.tree_map(jnp.add, acc["grads"], step_grads),
                "metrics": jax.tree_util.tree_map(jnp.add, acc["metrics"], step_aux),
            }
            return new_acc, step_aux

        start_indices = jnp.arange(num_accum_steps, dtype=jnp.int32) * minibatch_size
        final_acc, _aux = jax.lax.scan(
            accumulate_gradients,
            init_acc,
            start_indices,
            length=num_accum_steps,
        )
        gradients = jax.tree_util.tree_map(lambda x: x / num_accum_steps, final_acc["grads"])
        metrics = jax.tree_util.tree_map(lambda x: x / num_accum_steps, final_acc["metrics"])

    else:
        (_, metrics), gradients = grad_fn(state.graphstate, batch)

    return gradients, metrics  # type: ignore[return-value]
