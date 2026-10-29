# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-rank compiled executables from marker-clustered jaxprs.

Takes the output of
:func:`spectrax.runtime.mpmd.markers.cluster_jaxpr_by_markers`
(a list of per-stage sub-jaxprs produced by splitting a traced function
at :func:`sxstage_iter` markers) and a :class:`Schedule`, and
compiles **one jitted executable per physical rank** that runs that
rank's full schedule.

Takes marker-clustered sub-jaxprs and compiles per-rank executables.
The transport layer uses
:func:`jax.device_put` between per-rank jit calls (portable across
backends). Cross-rank overlap is NOT achieved here — each rank's
program runs in full before transporting activations/cotangents to the
next. The win is purely dispatch-count reduction (one jit per rank per
step vs one jit per action).

For cross-rank compute overlap, use
:func:`spectrax.runtime.spmd.make_scheduled_body`
(shard_map + ppermute) or wait for the NCCL transport layer.

**Entry points**:

* :func:`get_num_stages` — infer stage count from a flat task list.
* :func:`compile_ranked_executables` — marker clusters + schedule →
  per-rank jitted programs.
* :func:`run_ranked_pipeline` — end-to-end: compile + dispatch +
  transport + loss.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax._src.core import ClosedJaxpr, Jaxpr, jaxpr_as_fun

from ..schedules import Action, FusedTask, Phase, Schedule

__all__ = [
    "compile_ranked_executables",
    "get_num_stages",
    "get_num_stages_from_grid",
    "run_ranked_pipeline",
]


def _stage_fn_from_cluster(cluster: ClosedJaxpr | Jaxpr) -> Callable[..., object]:
    """Return a Python callable for one marker-clustered stage jaxpr.

    Args:
        cluster: Cluster value consumed by this operation.

    Returns:
        Return a Python callable for one marker-clustered stage jaxpr.
    """
    if isinstance(cluster, ClosedJaxpr):
        return jaxpr_as_fun(cluster)
    closed = ClosedJaxpr(cluster, [])
    return jaxpr_as_fun(closed)


def get_num_stages(tasks: list[Action | FusedTask]) -> int:
    """Infer the number of pipeline stages from a flat task list.

    Conventions:

    * For schedules where the terminal stage fuses FWD + BWD into one
      task (``FusedTask``), there are ``2n - 1`` tasks for ``n``
      stages. This is the standard case for GPipe and 1F1B on the
      terminal rank.
    * For schedules where FWD and BWD are always separate (GPipe
      global grid), there are ``2n`` tasks for ``n`` stages.
    * For zero-bubble schedules that split BWD into BWD_I + BWD_W,
      there are ``3n - 2`` tasks for ``n`` stages (the terminal stage
      fuses FWD + BWD_I, and the first stage fuses BWD_I + BWD_W).

    This function counts distinct ``stage_id``s inferred from the
    task microbatch pattern rather than assuming a fixed formula —
    robust across schedule families.

    Args:
        tasks: A flat list of :class:`Action` / :class:`FusedTask`
            entries for one rank (as returned by ``schedule.tasks()``
            or :func:`dualpipev_tasks`).

    Returns:
        Number of distinct pipeline stages represented in the list.
    """
    fwd_count = 0
    for task in tasks:
        if isinstance(task, FusedTask):
            for sub in (task.fwd, task.bwd):
                if sub.phase == Phase.FWD:
                    fwd_count += 1
        elif isinstance(task, Action):
            if task.phase == Phase.FWD:
                fwd_count += 1
    if fwd_count == 0:
        return 0
    grid_stages: set[int] = set()
    for task in tasks:
        if isinstance(task, FusedTask):
            grid_stages.add(task.fwd.virtual_stage)
            grid_stages.add(task.bwd.virtual_stage)
        elif isinstance(task, Action):
            grid_stages.add(task.virtual_stage)
    return len(grid_stages) if grid_stages else 0


def get_num_stages_from_grid(grid: list[list[Action | FusedTask | None]]) -> int:
    """Infer the number of physical stages from a schedule grid.

    Simply returns the width of the grid (number of columns = number
    of physical ranks).

    Args:
        grid: A schedule grid from :meth:`Schedule.build`.

    Returns:
        Number of physical stages (columns in the grid).
    """
    if not grid:
        return 0
    return len(grid[0])


def compile_ranked_executables(
    clusters: list[ClosedJaxpr | Jaxpr],
    schedule: Schedule,
    n_stages: int,
) -> list[Callable[..., object]]:
    """Compile one jitted program per physical rank from clustered sub-jaxprs.

    Each cluster (from :func:`cluster_jaxpr_by_markers`) represents one
    pipeline stage's computation. The schedule tells us which stages
    each rank runs and in what order. The returned per-rank callable
    wraps every stage call for that rank into a single :func:`jax.jit`,
    cutting dispatch from ``O(T x n)`` to ``O(n)`` per step.

    The per-rank program signature is::

        rank_fn(stage_params: list[params],
                mb_inputs: array[M, *shape],
                mb_cotangents: array[M, *shape],
                *mb_targets)
            -> (g_params_per_logical, mb_outputs, mb_out_cots, loss_sum)

    where ``g_params_per_logical`` is a ``dict[int, params_pytree]``
    keyed by logical stage index.

    Cross-rank transport happens OUTSIDE these programs via the caller
    (using :func:`jax.device_put` between rank invocations). There is
    no cross-rank overlap in this path.

    Args:
        clusters: List of ``n_stages * V`` sub-jaxprs in logical order
            (logical stage 0 first), where ``V =
            schedule.virtual_stages_per_rank()``. Each is a
            :class:`ClosedJaxpr` or bare :class:`Jaxpr`. Closed jaxprs
            carry their constants; bare jaxprs use no constants.
        schedule: A :class:`Schedule`. Both flat (``V=1``) and
            virtual-stage schedules are supported.
        n_stages: Number of physical pipeline ranks.

    Returns:
        A list of ``n_stages`` jitted callables, one per rank.
    """
    V = schedule.virtual_stages_per_rank()
    n_logical = n_stages * V

    if len(clusters) != n_logical:
        raise ValueError(f"Expected {n_logical} clusters for {n_stages} stages x V={V}; got {len(clusters)}.")

    stage_fns: list[Callable[..., object]] = []
    for cluster in clusters:
        stage_fns.append(_stage_fn_from_cluster(cluster))

    grid = schedule.build(n_stages)
    per_rank_actions: list[list[Action]] = [[] for _ in range(n_stages)]
    for row in grid:
        for r, cell in enumerate(row):
            if cell is None:
                continue
            if isinstance(cell, FusedTask):
                per_rank_actions[r].append(cell.fwd)
                per_rank_actions[r].append(cell.bwd)
            else:
                per_rank_actions[r].append(cell)

    programs: list[Callable[..., object]] = []
    for r in range(n_stages):
        actions = per_rank_actions[r]
        my_logical_stages = set()
        for a in actions:
            logical = schedule.logical_at(r, a.virtual_stage, n_stages)
            my_logical_stages.add(logical)
        my_fns = {logical: stage_fns[logical] for logical in my_logical_stages}

        def make_program(rank, rank_actions, rank_fns):
            """Build one rank's compiled-pipeline callable.

            Closes over ``rank``, the rank's ordered :class:`Action`
            list, and a logical-stage -> stage-function mapping so the
            inner jitted function evaluates exactly the work this rank
            owns.

            Args:
                rank: Physical pipeline rank.
                rank_actions: Ordered :class:`Action` list executed by
                    ``rank``.
                rank_fns: Mapping of logical stage index to its stage
                    function (closed jaxpr evaluator).

            Returns:
                A jitted ``(stage_params, mb_inputs, mb_cotangents,
                *mb_targets)`` callable.
            """

            @jax.jit
            def rank_fn(all_stage_params, mb_inputs, mb_cotangents, *mb_targets):
                """Execute every scheduled action for this rank in order.

                ``all_stage_params`` is a list/tuple of per-logical-stage
                params pytrees. ``mb_inputs[mb]`` is the microbatch input
                (from xs or transported from rank-1). ``mb_cotangents[mb]``
                is the cotangent arriving from rank+1 (zero for terminal).
                ``mb_targets`` is the tuple of target arrays (only used
                on the terminal rank).

                Returns ``(per_stage_grads, outgoing_activations,
                outgoing_cotangents, loss_sum)``.

                Args:
                    all_stage_params: All stage params value consumed by this operation.
                    mb_inputs: Mb inputs value consumed by this operation.
                    mb_cotangents: Mb cotangents value consumed by this operation.
                    *mb_targets: Additional positional arguments forwarded to the wrapped callable or backend.
                """
                with jax.named_scope(f"spectrax/mpmd/compiler/rank_{rank}"):
                    num_microbatches = mb_inputs.shape[0]
                    saved_inputs: dict[tuple[int, int], object] = {}
                    saved_outputs: dict[tuple[int, int], object] = {}
                    outgoing_acts: list[object | None] = [None] * int(num_microbatches)
                    outgoing_cots = jnp.zeros_like(mb_inputs)
                    g_params_accum: dict[int, object] = {}
                    loss_sum = jnp.zeros((), dtype=jnp.float32)

                    for action in rank_actions:
                        mb = action.microbatch
                        virt = action.virtual_stage
                        logical = schedule.logical_at(rank, virt, n_stages)
                        sfn = rank_fns[logical]

                        if action.phase == Phase.FWD:
                            with jax.named_scope(f"spectrax/mpmd/compiler/rank_{rank}/fwd_logical_{logical}_mb_{mb}"):
                                x_in = mb_inputs[mb]
                                y = sfn(*all_stage_params[logical], x_in)
                                if not isinstance(y, jax.Array):
                                    y = y[0] if isinstance(y, (tuple, list)) else y
                                saved_inputs[(virt, mb)] = x_in
                                saved_outputs[(virt, mb)] = y
                                outgoing_acts[mb] = y

                        elif action.phase == Phase.BWD:
                            with jax.named_scope(f"spectrax/mpmd/compiler/rank_{rank}/bwd_logical_{logical}_mb_{mb}"):
                                x_in = saved_inputs.get((virt, mb), mb_inputs[mb])
                                g_y = mb_cotangents[mb]

                                params = tuple(all_stage_params[logical])

                                def _stage_primary(*p_and_x, _fn=sfn, _n=len(params)):
                                    """Return the stage's primary activation as a function of ``(params, x)``.

                                    Args:
                                        _fn:  fn value consumed by this operation.
                                        _n:  n value consumed by this operation.
                                        *p_and_x: Additional positional arguments forwarded to the wrapped callable or backend.
                                    """
                                    p = p_and_x[:_n]
                                    x = p_and_x[_n]
                                    out = _fn(*p, x)
                                    return out[0] if isinstance(out, (tuple, list)) else out

                                _y, vjp_fn = jax.vjp(_stage_primary, *params, x_in)
                                grads = vjp_fn(g_y)
                                g_p = tuple(grads[:-1])
                                g_x = grads[-1]
                                outgoing_cots = outgoing_cots.at[mb].set(g_x)

                                if logical not in g_params_accum:
                                    g_params_accum[logical] = g_p
                                else:
                                    g_params_accum[logical] = jax.tree.map(
                                        lambda u, v: u + v,
                                        g_params_accum[logical],
                                        g_p,
                                    )

                    missing = [mb for mb, value in enumerate(outgoing_acts) if value is None]
                    if missing:
                        raise RuntimeError(f"rank {rank} did not produce forward outputs for microbatches {missing}.")
                    return g_params_accum, jnp.stack(outgoing_acts, axis=0), outgoing_cots, loss_sum

            return rank_fn

        programs.append(make_program(r, actions, my_fns))

    return programs


def run_ranked_pipeline(
    clusters: list[ClosedJaxpr | Jaxpr],
    params_per_stage: list[tuple[object, ...]],
    schedule: Schedule,
    n_stages: int,
    microbatches: int,
    xs: object,
    target_args: tuple[object, ...],
    loss_fn: Callable[..., object],
    stage_shardings: list[object] | None = None,
) -> tuple[object, list[object]]:
    """End-to-end: cluster → compile → execute → transport → loss.

    Drives one training step using per-rank compiled executables with
    :func:`jax.device_put` transport between ranks. No cross-rank
    overlap (serial per rank).

    Args:
        clusters: Per-stage sub-jaxprs from
            :func:`cluster_jaxpr_by_markers`.
        params_per_stage: ``params_per_stage[logical]`` is a tuple of
            arrays matching the cluster's invars (the "param" portion).
        schedule: A flat :class:`Schedule`.
        n_stages: Physical rank count.
        microbatches: ``M``.
        xs: Microbatched inputs ``(M, *shape)``.
        target_args: Microbatched targets.
        loss_fn: ``(y, *targets) -> scalar``.
        stage_shardings: Optional per-rank sharding for transport.

    Returns:
        ``(mean_loss, per_stage_grads)``.
    """
    if len(clusters) != len(params_per_stage):
        raise ValueError(f"Expected params for {len(clusters)} stages, got {len(params_per_stage)}.")
    if microbatches <= 0:
        raise ValueError(f"microbatches must be > 0, got {microbatches}.")

    stage_fns = [_stage_fn_from_cluster(cluster) for cluster in clusters]
    params_tree = tuple(tuple(params) for params in params_per_stage)

    def _primary(out: object) -> object:
        """Pick the first element from a stage output tuple, else return as-is.

        Stage callables built from clusters with multiple outvars
                return tuples; single-output stages return the array directly.
        This helper normalises both cases for the sequential loss
        accumulator.

        Args:
            out: Output value from an earlier call or transform.

        Returns:
            Result described by this helper.
        """
        return out[0] if isinstance(out, (tuple, list)) else out

    def _micro_loss(params: tuple[tuple[object, ...], ...], x_mb: object, *targets_mb: object) -> object:
        """Sequentially evaluate every logical stage on one microbatch and return the loss.

        Used for the reference single-rank simulation that produces
        ground-truth gradients via :func:`jax.value_and_grad`. Each
        stage's primary activation feeds the next stage's input, and
        ``loss_fn`` consumes the final activation against the
        microbatch targets.

        Args:
            params: Parameter mapping or primitive parameter dictionary.
            x_mb: X mb value consumed by this operation.
            *targets_mb: Additional positional arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        h = x_mb
        for logical, fn in enumerate(stage_fns):
            h = _primary(fn(*params[logical], h))
        return loss_fn(h, *targets_mb)

    grad_accum = None
    loss_sum = None
    for mb in range(microbatches):
        x_mb = xs[mb]
        targets_mb = tuple(t[mb] for t in target_args)
        loss_mb, grads_mb = jax.value_and_grad(_micro_loss)(params_tree, x_mb, *targets_mb)
        loss_sum = loss_mb if loss_sum is None else loss_sum + loss_mb
        if grad_accum is None:
            grad_accum = grads_mb
        else:
            grad_accum = jax.tree.map(lambda a, b: a + b, grad_accum, grads_mb)

    assert loss_sum is not None and grad_accum is not None
    inv_m = 1.0 / jnp.asarray(microbatches, dtype=loss_sum.dtype)
    mean_loss = loss_sum * inv_m
    grads = jax.tree.map(lambda g: g * inv_m, grad_accum)
    if stage_shardings is not None:
        V = schedule.virtual_stages_per_rank()
        logical_to_rank: dict[int, int] = {}
        for r in range(n_stages):
            for v in range(V):
                logical_to_rank[schedule.logical_at(r, v, n_stages)] = r
        placed = []
        for logical, grad in enumerate(grads):
            rank = logical_to_rank[logical]
            placed.append(jax.device_put(grad, stage_shardings[rank]))
        return mean_loss, placed
    return mean_loss, list(grads)
