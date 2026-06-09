# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fused value-and-grad-and-apply for schedule-driven MPMD pipelines.

Standard SpectraX usage today is a two-call dance: ``sxvalue_and_grad``
produces stage-local gradients, then the trainer separately applies its
optimizer update outside the MPMD schedule. That leaves the optimizer
step as a post-schedule barrier -- every rank has to wait for every
other rank's backward to finish before any apply can fire, even though
the apply on each rank is fully stage-local (its params, grads, and
optimizer state never leave that rank's submesh).

:func:`sxvalue_and_grad_and_apply` closes that gap. The returned
callable runs the same forward+backward schedule, but the optimizer
step is itself a *unit* in the schedule's dependency DAG: rank K's
apply unit depends only on rank K's last backward unit, so on schedules
where the rank-0 critical path dominates (chunked-KL distillation on a
tied-embed model is the canonical example), ranks 1-7 can run their
stage-local optimizer update **in parallel with rank-0's backward
tail**, instead of all eight ranks waiting at a global barrier.

Design notes (kept here so future readers don't have to reverse-engineer
the runtime):

* APPLY is a *runtime* concept, not a schedule concept. Schedules
  ``DualPipeV``, ``InterleavedH1``, etc.) emit only FWD/BWD/BWD_I/BWD_W
  cells. The schedule grid never contains APPLY entries.
  ``_build_schedule_units_from_plan`` synthesizes one APPLY unit per
  physical rank by checking ``plan["apply_jits"]`` -- when that map is
  populated by this module, apply units are emitted; otherwise the
  legacy ``sxvalue_and_grad`` path is byte-for-byte unchanged.
* The APPLY unit's dependency on "all of this rank's backward work" is
  expressed implicitly by the same-rank predecessor chain
  ``_build_schedule_unit_dependencies`` already builds for fwd/bwd
  units -- the chain naturally extends to APPLY units because they are
  appended after every real grid unit on their rank.
* The actual rank-local update is performed by a user-supplied
  ``apply_fn`` callable. SpectraX dispatches it under the rank's
  submesh context, hands it the rank index and a mutable apply context
  (containing ``grad_accums``, ``opt_state``, ``params``,
  ``learning_rate_fn``, output buffers, etc.), and the callable scatters
  the new params / new opt-state into the output buffers. Slicing the
  params / opt-state by rank ownership is the apply_fn's responsibility
  in this v1 -- a future refactor may move that slicing into SpectraX
  via ``leaf_stage_owners`` when there's a generic per-rank slicer for
  optax opt-state pytrees that doesn't constrain the optimizer choice.

What is **not** yet covered by v1:

* **Global gradient clipping** (e.g. ``clip_grad=1.0`` with
  ``clip_by_global_norm``). That needs a SCALAR_REDUCE schedule action
  to compute ``sqrt(sum_across_ranks(||g_local||^2))`` without forcing
  a full bwd barrier. Until that lands, callers that need global clip
  must either (a) compute it before calling this function and pass the
  pre-computed clip-scale into the apply_fn context, or (b) use
  stage-local clip, or (c) keep using the legacy
  ``sxvalue_and_grad`` + post-schedule apply path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import jax

from .runtime import (
    _arg_leaf_ranges,
    _dispatch_schedule_faithful,
    _ensure_schedule_plan,
    _normalize_argnums,
)


def sxvalue_and_grad_and_apply(
    fn: Callable,
    argnums: int | tuple[int, ...] = 0,
) -> Callable:
    """Fuse value-and-grad and optimizer-apply into one MPMD schedule pass.

    Like :func:`sxvalue_and_grad`, this wraps an ``sxjit``-decorated
    function whose body produces a scalar loss. Unlike
    :func:`sxvalue_and_grad`, the returned callable also runs the
    optimizer update inline -- as per-rank APPLY units in the schedule's
    dependency DAG -- so rank K's apply can fire while other ranks are
    still finishing backward, instead of waiting at a global barrier
    outside the runtime.

    Args:
        fn: ``sxjit``-decorated function whose body emits ``sxstage_iter``
            (or ``sxstage_region``) boundaries and returns a scalar loss.
        argnums: Index (or indices) of arguments to differentiate with
            respect to. Default ``0`` (first argument is the params).

    Returns:
        A callable. Its signature is::

            vga_fn(
                *args,                       # same positional args as ``fn``
                apply_fn,                    # required: per-rank apply callable
                opt_state,                   # required: optimizer state pytree
                learning_rate_fn=None,       # optional schedule callback
                **apply_context,             # opaque -- forwarded to ``apply_fn``
            ) -> (loss, new_params, new_opt_state)

        ``apply_fn`` is called once per physical rank with signature
        ``apply_fn(rank: int, *, grad_accums: dict[int, jax.Array],
        state: dict)`` and must write the updated rank-local params and
        opt-state into ``state["new_params_buf"]`` and
        ``state["new_opt_state_buf"]``. The runtime gathers those buffers
        once every rank's apply unit has completed and returns the
        assembled new_params + new_opt_state to the caller.

    Raises:
        TypeError: If ``fn`` is not ``sxjit``-decorated with a schedule.
        RuntimeError: If an apply unit fires without a matching apply
            context (a logic bug in the caller -- shouldn't happen via
            the public API).
    """
    if not hasattr(fn, "_mpmd_state") or not fn._mpmd_state.get("schedule_requested", False):
        raise TypeError("sxvalue_and_grad_and_apply requires an sxjit-decorated function with a schedule.")

    if isinstance(argnums, int):
        norm_argnums: tuple[int, ...] = (argnums,)
    else:
        norm_argnums = tuple(argnums)

    def vga_fn(
        *args: object,
        apply_fn: Callable[..., None],
        opt_state: Any,
        learning_rate_fn: Any = None,
        **apply_context_extras: Any,
    ) -> tuple[jax.Array, Any, Any]:
        validated_argnums = _normalize_argnums(norm_argnums, len(args))
        plan = _ensure_schedule_plan(fn, args, grad_argnums=validated_argnums)

        primary_argnum = validated_argnums[0]
        params = args[primary_argnum]
        leaf_stage_owners = plan.get("leaf_stage_owners", {})
        rank_submeshes = plan.get("rank_submeshes", ())
        apply_context: dict[str, Any] = {
            "apply_fn": apply_fn,
            "params": params,
            "opt_state": opt_state,
            "learning_rate_fn": learning_rate_fn,
            "leaf_stage_owners": leaf_stage_owners,
            "rank_submeshes": rank_submeshes,
            "new_params_buf": {},
            "new_opt_state_buf": {},
            **apply_context_extras,
        }

        n_rank = int(plan.get("n", 0))
        plan["apply_jits"] = {(rank, 0): True for rank in range(n_rank)}
        plan["apply_context"] = apply_context
        plan["apply_requires_all_grads"] = bool(apply_context_extras.get("apply_requires_all_grads", False))

        try:
            loss, grads_flat = _dispatch_schedule_faithful(plan, args, return_loss=True)
        finally:
            plan.pop("apply_context", None)
            plan.pop("apply_jits", None)
            plan.pop("apply_requires_all_grads", None)

        leaf_ranges = _arg_leaf_ranges(args)
        grads_for_log: list[object] = []
        for argnum in validated_argnums:
            start, end = leaf_ranges[argnum]
            arg_leaves = grads_flat[start:end]
            if len(arg_leaves) == 1:
                grads_for_log.append(arg_leaves[0])
            else:
                grads_for_log.append(jax.tree.unflatten(jax.tree.structure(args[argnum]), list(arg_leaves)))
        apply_context["grads_for_log"] = tuple(grads_for_log)

        assemble = apply_context.get("assemble_outputs")
        if callable(assemble):
            new_params, new_opt_state = assemble(apply_context)
        else:
            new_params, new_opt_state = _default_assemble_outputs(apply_context)

        return cast(jax.Array, loss), new_params, new_opt_state

    return vga_fn


def _default_assemble_outputs(apply_context: dict[str, Any]) -> tuple[Any, Any]:
    """Default assembler: last rank to write wins.

    Works when ``apply_fn`` writes the FULL params / opt-state tree each
    time, with only its rank's leaves updated (other leaves left as the
    original input values). This is the simplest correct contract for v1
    -- the apply work is still rank-local because the kernel only touches
    rank-owned leaves; the cross-rank "merge" is just a no-op overwrite
    of identical values.

    For larger param counts (where re-writing the full tree per rank is
    wasteful), callers should provide a custom ``assemble_outputs``
    via the ``apply_context_extras`` kwarg that constructs the final
    tree from per-rank slices.

    Args:
        apply_context: The apply context dict the runtime threaded through
            apply_fn calls.

    Returns:
        ``(new_params, new_opt_state)`` -- the assembled outputs.

    Raises:
        RuntimeError: If no rank wrote into ``new_params_buf`` /
            ``new_opt_state_buf`` (the apply_fn never fired, which would
            be a bug).
    """
    new_params_buf = apply_context["new_params_buf"]
    new_opt_state_buf = apply_context["new_opt_state_buf"]
    if not new_params_buf or not new_opt_state_buf:
        raise RuntimeError(
            "SpectraX sxvalue_and_grad_and_apply: no rank wrote into the apply output buffers. "
            "This usually means the apply_fn was never invoked -- check that the schedule has "
            "enough physical ranks to host the model and that apply_jits got populated."
        )
    last_rank = max(new_params_buf.keys())
    return new_params_buf[last_rank], new_opt_state_buf[last_rank]
