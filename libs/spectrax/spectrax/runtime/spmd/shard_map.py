# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":func:`make_scheduled_body` — shard_map body for arbitrary pipeline schedules.

Turns any :class:`~spectrax.runtime.schedules.Schedule`'s action grid
into a :func:`jax.shard_map` body with the ``pp`` axis **manual** and
cross-rank transport via :func:`jax.lax.ppermute`.

**Two compilation modes**:

* **Python-unrolled** (default when ``use_scan=False``): each time step
  is a separate HLO block with its own ``lax.cond`` + ``ppermute``.
  Fast compile at small scale; OOMs at 8B+ with virtual stages because
  the HLO graph grows linearly with ``T x n_actions``.

* **Scan-based** (``use_scan=True``): the schedule is encoded as
  integer arrays and ``lax.scan`` loops over time steps. ONE copy of
  the body in HLO regardless of ``T`` → compiles at any scale. Dynamic
  dispatch inside the scan via ``lax.switch`` over phase x ``lax.cond``
  over rank. No ``value_and_grad`` over the scan (explicit ``fwd_fn``
  + ``bwd_fn`` avoids the autograd-through-scan memory blowup).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp

from ..schedules import Action, FusedTask, Phase, Schedule

__all__ = ["make_scheduled_body"]

_PHASE_SKIP = 0
_PHASE_FWD = 1
_PHASE_BWD = 2
_PHASE_BWD_I = 3
_PHASE_BWD_W = 4

_PHASE_MAP = {
    Phase.FWD: _PHASE_FWD,
    Phase.BWD: _PHASE_BWD,
    Phase.BWD_I: _PHASE_BWD_I,
    Phase.BWD_W: _PHASE_BWD_W,
}


def _drop0(tree: object) -> object:
    """Strip the leading size-1 pp axis from every leaf in ``tree``.

    Args:
        tree: object pytree of arrays.

    Returns:
        Same pytree with each leaf indexed at ``[0]``.
    """
    return jax.tree.map(lambda a: a[0], tree)


def _add0(tree: object) -> object:
    """Add a leading size-1 pp axis to every leaf in ``tree``.

    Inverse of :func:`_drop0`; used on out-bound carries that need
    to satisfy a ``PartitionSpec(pp_axis)`` ``out_specs`` entry.

    Args:
        tree: object pytree of arrays.

    Returns:
        Same pytree with each leaf gaining a leading axis of size 1.
    """
    return jax.tree.map(lambda a: a[None, ...], tree)


def _prev_logical_loc(schedule: Schedule, rank: int, virt: int, n_stages: int) -> tuple[int, int] | None:
    """Return the ``(rank, virt)`` hosting logical stage ``logical - 1``.

    The schedule's :meth:`Schedule.next_logical_loc` answers the
    forward direction; this helper inverse-searches it so the runtime
    knows where to ``ppermute`` a backward cotangent. Search is
    O(n_stages * V) but only runs at trace time, not per step.

    Args:
        schedule: The active :class:`Schedule`.
        rank: Current physical rank.
        virt: Current virtual-stage index.
        n_stages: Number of physical pipeline ranks.

    Returns:
        ``(rank, virt)`` of the upstream logical stage, or ``None``
        if this is the first logical stage (no upstream).
    """
    logical = schedule.logical_at(rank, virt, n_stages)
    if logical == 0:
        return None
    V = schedule.virtual_stages_per_rank()
    for r in range(n_stages):
        for v in range(V):
            if schedule.logical_at(r, v, n_stages) == logical - 1:
                return (r, v)
    return None


def _encode_grid(
    schedule: Schedule, n_stages: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Encode the schedule grid as integer arrays for :func:`lax.scan`.

    The scan-based body cannot iterate Python objects, so the schedule
    grid (a list of :class:`Action` cells) is flattened into integer
    arrays of shape ``(T, n_stages)`` indexed by time step. The body
    then reads its rank's column at each scan step and dispatches via
    :func:`jax.lax.switch`.

    ``fwd_dest_grid[t, r]`` is the physical rank that rank ``r``'s
    FWD output should be transported to at time ``t`` (``-1`` if no
    transport, i.e. the action is terminal or non-FWD).
    ``bwd_dest_grid[t, r]`` is where the BWD cotangent goes (the
    upstream rank, ``-1`` if first logical stage or non-BWD).
    ``fwd_dv_grid`` / ``bwd_dv_grid`` carry the virtual-stage index
    of the destination (used to write into the right slot of the
    receiver's per-virt buffer).

    Args:
        schedule: The schedule to encode.
        n_stages: Number of physical pipeline ranks.

    Returns:
        ``(phase_grid, mb_grid, virt_grid, logical_grid,
        fwd_dest_grid, fwd_dv_grid, bwd_dest_grid, bwd_dv_grid, T)``.
    """
    grid_raw = schedule.build(n_stages)
    T = len(grid_raw)
    phase_arr = [[_PHASE_SKIP] * n_stages for _ in range(T)]
    mb_arr = [[0] * n_stages for _ in range(T)]
    virt_arr = [[0] * n_stages for _ in range(T)]
    logical_arr = [[-1] * n_stages for _ in range(T)]
    fwd_dest_arr = [[-1] * n_stages for _ in range(T)]
    fwd_dest_virt_arr = [[0] * n_stages for _ in range(T)]
    bwd_dest_arr = [[-1] * n_stages for _ in range(T)]
    bwd_dest_virt_arr = [[0] * n_stages for _ in range(T)]
    for t, row in enumerate(grid_raw):
        for r, cell in enumerate(row):
            if cell is None:
                continue
            if isinstance(cell, FusedTask):
                cell = cell.fwd
            phase_arr[t][r] = _PHASE_MAP.get(cell.phase, _PHASE_SKIP)
            mb_arr[t][r] = cell.microbatch
            virt_arr[t][r] = cell.virtual_stage
            v = cell.virtual_stage
            logical_arr[t][r] = schedule.logical_at(r, v, n_stages)
            if cell.phase == Phase.FWD:
                nxt = schedule.next_logical_loc(r, v, n_stages)
                if nxt is not None:
                    fwd_dest_arr[t][r] = nxt[0]
                    fwd_dest_virt_arr[t][r] = nxt[1]
            elif cell.phase in (Phase.BWD, Phase.BWD_I):
                prev = _prev_logical_loc(schedule, r, v, n_stages)
                if prev is not None:
                    bwd_dest_arr[t][r] = prev[0]
                    bwd_dest_virt_arr[t][r] = prev[1]
    return (
        jnp.array(phase_arr, dtype=jnp.int32),
        jnp.array(mb_arr, dtype=jnp.int32),
        jnp.array(virt_arr, dtype=jnp.int32),
        jnp.array(logical_arr, dtype=jnp.int32),
        jnp.array(fwd_dest_arr, dtype=jnp.int32),
        jnp.array(fwd_dest_virt_arr, dtype=jnp.int32),
        jnp.array(bwd_dest_arr, dtype=jnp.int32),
        jnp.array(bwd_dest_virt_arr, dtype=jnp.int32),
        T,
    )


def make_scheduled_body(
    *,
    schedule: Schedule,
    n_stages: int,
    microbatches: int,
    pp_axis: str,
    fwd_fn: Callable[[object]],
    bwd_fn: Callable[[object], tuple[object]],
    loss_and_g_y: Callable[..., tuple[object]],
    mode: Literal["train"] = "train",
    checkpoint_policy: Callable[..., bool] | None = None,
    use_scan: bool = False,
) -> Callable[..., object]:
    """Build a :func:`jax.shard_map` body that executes ``schedule``.

    Args:
            schedule: Flat or virtual-stage :class:`Schedule`.
            n_stages: Physical pipeline rank count.
            microbatches: ``M``.
            pp_axis: Manual pipeline-parallel mesh axis name.
            fwd_fn: ``(params, x) -> y`` per microbatch.
            bwd_fn: ``(params, x, g_y) -> (g_params, g_x)`` per microbatch.
            loss_and_g_y: ``(y, *targets) -> (loss, g_y)`` terminal rank.
            mode: Only ``"train"`` supported.
            checkpoint_policy: If truthy, wrap fwd/bwd in ``jax.checkpoint``.
            use_scan: Use ``lax.scan`` (compact HLO, scales to 8B+) or
                Python-unrolled (small-scale only). Default ``False``.

    Returns:
        Result described by this helper.
    """
    if mode != "train":
        raise NotImplementedError(f"make_scheduled_body only supports mode='train'; got {mode}.")

    V = schedule.virtual_stages_per_rank()
    n_logical = n_stages * V
    m = microbatches

    if checkpoint_policy is not None:
        _raw_fwd, _raw_bwd = fwd_fn, bwd_fn

        def fwd_fn(params, x):
            """Forward wrapped in :func:`jax.checkpoint` for activation rematerialisation.

            Trades compute for memory: the forward's intermediate
            activations are recomputed during backward instead of
            saved.

            Args:
                params: Stage parameters.
                x: Stage input.

            Returns:
                Stage output ``y``.
            """

            @jax.checkpoint
            def _ckpt(p, xi):
                """Inner checkpoint boundary marking ``_raw_fwd`` for rematerialisation.

                Args:
                    p: P value consumed by this operation.
                    xi: Xi value consumed by this operation.
                """
                return _raw_fwd(p, xi)

            return _ckpt(params, x)

        def bwd_fn(params, x, g_y):
            """Backward wrapped in :func:`jax.checkpoint` for higher-order remat.

            Args:
                params: Stage parameters.
                x: Saved input.
                g_y: Output cotangent.

            Returns:
                ``(g_params, g_x)``.
            """

            @jax.checkpoint
            def _ckpt(p, xi, g):
                """Inner checkpoint boundary for ``_raw_bwd`` under higher-order autodiff.

                Args:
                    p: P value consumed by this operation.
                    xi: Xi value consumed by this operation.
                    g: G value consumed by this operation.
                """
                return _raw_bwd(p, xi, g)

            return _ckpt(params, x, g_y)

    if use_scan:
        return _make_scan_body(
            schedule=schedule,
            n_stages=n_stages,
            microbatches=m,
            pp_axis=pp_axis,
            fwd_fn=fwd_fn,
            bwd_fn=bwd_fn,
            loss_and_g_y=loss_and_g_y,
            V=V,
            n_logical=n_logical,
        )
    return _make_unrolled_body(
        schedule=schedule,
        n_stages=n_stages,
        microbatches=m,
        pp_axis=pp_axis,
        fwd_fn=fwd_fn,
        bwd_fn=bwd_fn,
        loss_and_g_y=loss_and_g_y,
        V=V,
        n_logical=n_logical,
    )


def _make_scan_body(
    *,
    schedule,
    n_stages,
    microbatches,
    pp_axis,
    fwd_fn,
    bwd_fn,
    loss_and_g_y,
    V,
    n_logical,
):
    """Scan-based body: :func:`jax.lax.scan` over time steps with dynamic dispatch.

    Compiles to ONE copy of the step body in HLO regardless of how
    many time steps the schedule has. The schedule is encoded into
    integer arrays by :func:`_encode_grid`; the body reads its rank's
    column at each scan iteration, dispatches via
    :func:`jax.lax.switch` over the action's phase, and routes
    activations / cotangents via :func:`jax.lax.ppermute` over all
    rank-to-rank pairs (only the matching pair carries non-junk data,
    selected via ``where``).

    Args:
        schedule: A :class:`Schedule`.
        n_stages: Number of physical pipeline ranks.
        microbatches: ``M``.
        pp_axis: Manual pipeline-parallel mesh axis name.
        fwd_fn: ``(params, x) -> y``.
        bwd_fn: ``(params, x, g_y) -> (g_params, g_x)``.
        loss_and_g_y: ``(y, *targets) -> (loss, g_y)`` for the
            terminal stage.
        V: Virtual stages per rank.
        n_logical: Total logical stages (``n_stages * V``).

    Returns:
        A ``shard_map`` body function.
    """
    (
        phase_grid,
        mb_grid,
        virt_grid,
        logical_grid,
        fwd_dest_grid,
        fwd_dv_grid,
        bwd_dest_grid,
        bwd_dv_grid,
        T,
    ) = _encode_grid(schedule, n_stages)
    m = microbatches

    def body(stacked_params, xs, *targets):
        """Shard-map body: scan over schedule time steps.

        Args:
            stacked_params: Per-rank stacked parameters.
            xs: Microbatched input ``(M, ...)``.
            *targets: Microbatched targets.

        Returns:
            ``(mean_loss, stacked_params_grads)``.
        """
        rank = jax.lax.axis_index(pp_axis)
        p_per_rank = _drop0(stacked_params)
        mb_shape = xs.shape[1:]
        mb_dtype = xs.dtype

        def pick_virt(tree, v):
            """Select the virtual-stage slice ``v`` from per-rank parameters.

            For flat schedules (``V == 1``) the params have no virt
            axis and the tree is returned as-is.

            Args:
                tree: Per-rank parameter pytree.
                v: Virtual-stage index.

            Returns:
                The virt slice of every leaf.
            """
            if V == 1:
                return tree
            return jax.tree.map(lambda a: a[v], tree)

        if V == 1:
            saved_inputs = jnp.zeros((m, *mb_shape), dtype=mb_dtype)
            saved_outputs = jnp.zeros((m, *mb_shape), dtype=mb_dtype)
            incoming_fwd = jnp.zeros((m, *mb_shape), dtype=mb_dtype)
            incoming_bwd = jnp.zeros((m, *mb_shape), dtype=mb_dtype)
        else:
            saved_inputs = jnp.zeros((V, m, *mb_shape), dtype=mb_dtype)
            saved_outputs = jnp.zeros((V, m, *mb_shape), dtype=mb_dtype)
            incoming_fwd = jnp.zeros((V, m, *mb_shape), dtype=mb_dtype)
            incoming_bwd = jnp.zeros((V, m, *mb_shape), dtype=mb_dtype)

        g_params = jax.tree.map(jnp.zeros_like, p_per_rank)
        loss_acc = jnp.zeros((), dtype=jnp.float32)

        my_phases = phase_grid[:, rank]
        my_mbs = mb_grid[:, rank]
        my_virts = virt_grid[:, rank]
        my_logicals = logical_grid[:, rank]
        my_fwd_dests = fwd_dest_grid[:, rank]
        my_bwd_dests = bwd_dest_grid[:, rank]

        def si_get(arr, v, mb_idx):
            """Read ``arr[v, mb_idx]`` or ``arr[mb_idx]`` depending on ``V``.

            Args:
                arr: Saved-input / saved-output / incoming buffer.
                v: Virtual-stage index (ignored when ``V == 1``).
                mb_idx: Microbatch index.

            Returns:
                The buffered value for this ``(v, mb)`` slot.
            """
            return arr[mb_idx] if V == 1 else arr[v, mb_idx]

        def si_set(arr, v, mb_idx, val):
            """Functional write of ``val`` into ``arr[v, mb_idx]`` (or ``arr[mb_idx]``).

            Args:
                arr: Buffer to write into.
                v: Virtual-stage index (ignored when ``V == 1``).
                mb_idx: Microbatch index.
                val: Value to store.

            Returns:
                The updated buffer.
            """
            return arr.at[mb_idx].set(val) if V == 1 else arr.at[v, mb_idx].set(val)

        def g_add(acc, v, upd):
            """Accumulate ``upd`` into the grad accumulator at virt ``v``.

            Args:
                acc: Grad accumulator pytree.
                v: Virtual-stage index (ignored when ``V == 1``).
                upd: Per-leaf gradient update.

            Returns:
                Updated accumulator pytree.
            """
            if V == 1:
                return jax.tree.map(lambda a, b: a + b, acc, upd)
            return jax.tree.map(lambda a, b: a.at[v].add(b), acc, upd)

        carry_init = (saved_inputs, saved_outputs, incoming_fwd, incoming_bwd, g_params, loss_acc)

        transfer_edges = tuple((src, dst) for src in range(n_stages) for dst in range(n_stages) if src != dst)

        def route_value(buffer, value, dest, dest_virt, mb_idx):
            """Route ``value`` from this rank to ``dest`` and store it in ``buffer[dest_virt, mb_idx]``.

            Same-rank routing is a direct buffer write. Cross-rank
            routing fires :func:`jax.lax.ppermute` for every possible
            ``(src, dst)`` pair (collectives must run on every device
            every scan step) and uses the ``mb`` tag to select which
            transfer actually carries data — the rest are masked out
            by the ``has_value`` predicate.

            Args:
                buffer: Receiver-side buffer to write into.
                value: Value being sent.
                dest: Destination physical rank.
                dest_virt: Destination virtual-stage index.
                mb_idx: Microbatch index for both buffer slot and
                    routing tag.

            Returns:
                The updated buffer (logically a no-op on ranks that
                are neither sender nor receiver for this transfer).
            """
            same_rank = dest == rank
            buffer = jnp.where(same_rank, si_set(buffer, dest_virt, mb_idx, value), buffer)
            for src, dst in transfer_edges:
                send = (rank == src) & (dest == dst)
                mb_tag = jnp.where(send, mb_idx, -1).astype(jnp.int32)
                v_tag = jnp.where(send, dest_virt, 0).astype(jnp.int32)
                recv_value = jax.lax.ppermute(value, pp_axis, perm=[(src, dst)])
                recv_mb = jax.lax.ppermute(mb_tag, pp_axis, perm=[(src, dst)])
                recv_v = jax.lax.ppermute(v_tag, pp_axis, perm=[(src, dst)])
                has_value = (rank == dst) & (recv_mb >= 0)
                buffer = jnp.where(has_value, si_set(buffer, recv_v, recv_mb, recv_value), buffer)
            return buffer

        def scan_step(carry, t_idx):
            """One scan iteration: dispatch the action then transport its outputs.

            Each iteration reads its rank's slice of the integer
            grids at ``t_idx``, picks the matching virtual stage's
            params, and dispatches to one of the do_* phase branches
            via :func:`jax.lax.switch`. The branch returns the
            forward / backward value to transport (zeros for skip /
            non-applicable phases); transport via :func:`route_value`
            then runs unconditionally because collectives can't live
            inside ``lax.switch`` branches (only one branch executes
            per device, but collectives need every device).

            Args:
                carry: Tuple ``(saved_inputs, saved_outputs,
                    incoming_fwd, incoming_bwd, g_params, loss_acc)``.
                t_idx: Scan time-step index.

            Returns:
                ``(new_carry, None)`` — scan emits no per-step output.
            """
            si, so, ifwd, ibwd, gp, la = carry
            phase = my_phases[t_idx]
            mb_idx = my_mbs[t_idx]
            v = my_virts[t_idx]
            p_v = pick_virt(p_per_rank, v)

            logical = my_logicals[t_idx]
            is_first_logical = logical == 0
            is_terminal = logical == n_logical - 1

            x_from_xs = xs[mb_idx]
            x_from_incoming = si_get(ifwd, v, mb_idx)
            x_in = jnp.where(is_first_logical, x_from_xs, x_from_incoming)

            def do_fwd(args):
                """FWD branch: compute ``y``, save ``x_in`` (and ``y`` if terminal), emit ``y`` for transport.

                Args:
                    args: ``(saved_inputs, saved_outputs, g_params, loss_acc)``.

                Returns:
                    Updated ``(si, so, gp, la)`` plus ``(fwd_val,
                    bwd_val)`` where ``fwd_val == y`` and
                    ``bwd_val == 0`` (no backward transport).
                """
                si_, so_, gp_, la_ = args
                y = fwd_fn(p_v, x_in)
                si_ = si_set(si_, v, mb_idx, x_in)
                so_ = jnp.where(is_terminal, si_set(so_, v, mb_idx, y), so_)
                return si_, so_, gp_, la_, y, jnp.zeros_like(y)

            def do_bwd(args):
                """Full BWD branch: accumulate ``g_params``, emit ``g_x`` for transport upstream.

                Args:
                    args: ``(saved_inputs, saved_outputs, g_params, loss_acc)``.

                Returns:
                    Updated ``(si, so, gp, la)`` plus ``(0, g_x)`` —
                    nothing flows downstream, ``g_x`` flows
                    upstream.
                """
                si_, so_, gp_, la_ = args
                x_saved = si_get(si_, v, mb_idx)

                def _get_gy():
                    """Terminal-stage branch: compute ``(loss, g_y)`` from saved output and targets."""
                    y_out = si_get(so_, v, mb_idx)
                    loss_mb, g_y_mb = loss_and_g_y(y_out, *(t_arr[mb_idx] for t_arr in targets))
                    return loss_mb.astype(jnp.float32), g_y_mb

                def _get_gy_incoming():
                    """Non-terminal branch: ``loss = 0``, ``g_y`` = previously received cotangent buffer entry."""
                    return jnp.zeros((), dtype=jnp.float32), si_get(ibwd, v, mb_idx)

                loss_mb, g_y = jax.lax.cond(is_terminal, _get_gy, _get_gy_incoming)
                la_ = la_ + loss_mb
                g_p, g_x = bwd_fn(p_v, x_saved, g_y)
                gp_ = g_add(gp_, v, g_p)
                return si_, so_, gp_, la_, jnp.zeros_like(x_saved), g_x

            def do_bwd_i(args):
                """Input-gradient half of zero-bubble: emit ``g_x`` upstream, do not accumulate ``g_p``.

                Used by :class:`ZeroBubbleH1` so the critical-path
                input-grad fires immediately while the weight-grad is
                deferred.

                Args:
                    args: ``(saved_inputs, saved_outputs, g_params, loss_acc)``.

                Returns:
                    Updated carry plus ``(0, g_x)`` — only upstream
                    transport.
                """
                si_, so_, gp_, la_ = args
                x_saved = si_get(si_, v, mb_idx)

                def _get_gy():
                    """Terminal-stage branch: ``(loss, g_y)`` from saved output and targets."""
                    y_out = si_get(so_, v, mb_idx)
                    loss_mb, g_y_mb = loss_and_g_y(y_out, *(t_arr[mb_idx] for t_arr in targets))
                    return loss_mb.astype(jnp.float32), g_y_mb

                def _get_gy_incoming():
                    """Non-terminal branch: ``loss = 0``, ``g_y`` from incoming cotangent buffer."""
                    return jnp.zeros((), dtype=jnp.float32), si_get(ibwd, v, mb_idx)

                loss_mb, g_y = jax.lax.cond(is_terminal, _get_gy, _get_gy_incoming)
                la_ = la_ + loss_mb
                _g_p, g_x = bwd_fn(p_v, x_saved, g_y)
                return si_, so_, gp_, la_, jnp.zeros_like(x_saved), g_x

            def do_bwd_w(args):
                """Weight-gradient half of zero-bubble: accumulate ``g_p``, no upstream send.

                Args:
                    args: ``(saved_inputs, saved_outputs, g_params, loss_acc)``.

                Returns:
                    Updated carry plus ``(0, 0)`` — nothing
                    transports out of a weight-grad action.
                """
                si_, so_, gp_, la_ = args
                x_saved = si_get(si_, v, mb_idx)

                def _get_gy():
                    """Terminal-stage branch: pull ``g_y`` from loss; loss is intentionally not added (BWD_I already did)."""
                    y_out = si_get(so_, v, mb_idx)
                    _loss_mb, g_y_mb = loss_and_g_y(y_out, *(t_arr[mb_idx] for t_arr in targets))
                    return g_y_mb

                def _get_gy_incoming():
                    """Non-terminal branch: pull ``g_y`` from the incoming cotangent buffer."""
                    return si_get(ibwd, v, mb_idx)

                g_y = jax.lax.cond(is_terminal, _get_gy, _get_gy_incoming)
                g_p, _g_x = bwd_fn(p_v, x_saved, g_y)
                gp_ = g_add(gp_, v, g_p)
                return si_, so_, gp_, la_, jnp.zeros_like(x_saved), jnp.zeros_like(x_saved)

            def do_skip(args):
                """Idle branch: carry unchanged, transport zeros (masked out by ``has_value``).

                Args:
                    args: ``(saved_inputs, saved_outputs, g_params, loss_acc)``.

                Returns:
                    Carry unchanged plus ``(0, 0)`` for transport.
                """
                si_, so_, gp_, la_ = args
                z = jnp.zeros(mb_shape, dtype=mb_dtype)
                return si_, so_, gp_, la_, z, z

            args = (si, so, gp, la)
            si_, so_, gp_, la_, fwd_val, bwd_val = jax.lax.switch(
                phase,
                [do_skip, do_fwd, do_bwd, do_bwd_i, do_bwd_w],
                args,
            )

            fwd_dest = my_fwd_dests[t_idx]
            fwd_dv = fwd_dv_grid[:, rank][t_idx]
            bwd_dest = my_bwd_dests[t_idx]
            bwd_dv = bwd_dv_grid[:, rank][t_idx]

            ifwd_ = route_value(ifwd, fwd_val, fwd_dest, fwd_dv, mb_idx)
            ibwd_ = route_value(ibwd, bwd_val, bwd_dest, bwd_dv, mb_idx)

            return (si_, so_, ifwd_, ibwd_, gp_, la_), None

        (_si_f, _so_f, _ifwd_f, _ibwd_f, gp_f, la_f), _ = jax.lax.scan(scan_step, carry_init, jnp.arange(T))

        total_loss = jax.lax.psum(la_f, pp_axis)
        mean_loss = total_loss / jnp.asarray(m, dtype=total_loss.dtype)
        return mean_loss, _add0(gp_f)

    return body


def _make_unrolled_body(
    *,
    schedule,
    n_stages,
    microbatches,
    pp_axis,
    fwd_fn,
    bwd_fn,
    loss_and_g_y,
    V,
    n_logical,
):
    """Python-unrolled ``shard_map`` body: one HLO block per schedule cell.

    The classic implementation: walk the schedule grid in Python and
    emit one block of XLA ops per ``(t, rank)`` cell, using
    :func:`jax.lax.cond` to gate the active rank and
    :func:`jax.lax.ppermute` for cross-rank transport. Compiles fast
    at small scale and is straightforward to debug, but the HLO
    graph grows as ``T * n_actions`` so OOMs on 8B+ pipelines with
    virtual stages — use ``use_scan=True`` for those.

    Args:
        schedule: A :class:`Schedule`.
        n_stages: Number of physical pipeline ranks.
        microbatches: ``M``.
        pp_axis: Manual pipeline-parallel mesh axis name.
        fwd_fn: ``(params, x) -> y``.
        bwd_fn: ``(params, x, g_y) -> (g_params, g_x)``.
        loss_and_g_y: Terminal ``(y, *targets) -> (loss, g_y)``.
        V: Virtual stages per rank.
        n_logical: Total logical stages.

    Returns:
        A ``shard_map`` body function.
    """
    m = microbatches

    grid_raw = schedule.build(n_stages)
    grid: list[list[Action | None]] = []
    for row in grid_raw:
        new_row: list[Action | None] = []
        for cell in row:
            if cell is None:
                new_row.append(None)
            elif isinstance(cell, FusedTask):
                new_row.append(cell.fwd)
                new_row.append(cell.bwd)
            else:
                new_row.append(cell)
        grid.append(new_row[:n_stages])

    def body(stacked_params, xs, *targets):
        """Shard-map body: walk the schedule grid as Python-unrolled time steps.

        Args:
            stacked_params: Per-rank stacked params.
            xs: Microbatched inputs ``(M, ...)``.
            *targets: Microbatched targets.

        Returns:
            ``(mean_loss, stacked_params_grads)``.
        """
        rank = jax.lax.axis_index(pp_axis)
        p_per_rank = _drop0(stacked_params)
        mb_shape = xs.shape[1:]
        mb_dtype = xs.dtype

        def pick_virt(tree, v):
            """Select the virtual-stage slice ``v`` from per-rank parameters.

            Args:
                tree: Per-rank parameter pytree.
                v: Virtual-stage index (ignored when ``V == 1``).

            Returns:
                The virt slice of every leaf, or ``tree`` unchanged
                for flat schedules.
            """
            if V == 1:
                return tree
            return jax.tree.map(lambda a: a[v], tree)

        if V == 1:
            saved_inputs = jnp.zeros((m, *mb_shape), dtype=mb_dtype)
            saved_outputs = jnp.zeros((m, *mb_shape), dtype=mb_dtype)
            incoming_fwd = jnp.zeros((m, *mb_shape), dtype=mb_dtype)
            incoming_bwd = jnp.zeros((m, *mb_shape), dtype=mb_dtype)
        else:
            saved_inputs = jnp.zeros((V, m, *mb_shape), dtype=mb_dtype)
            saved_outputs = jnp.zeros((V, m, *mb_shape), dtype=mb_dtype)
            incoming_fwd = jnp.zeros((V, m, *mb_shape), dtype=mb_dtype)
            incoming_bwd = jnp.zeros((V, m, *mb_shape), dtype=mb_dtype)

        g_params = jax.tree.map(jnp.zeros_like, p_per_rank)
        loss_acc = jnp.zeros((), dtype=jnp.float32)

        def si_get(arr, v, mb_idx):
            """Read ``arr[v, mb_idx]`` (or ``arr[mb_idx]`` when ``V == 1``).

            Args:
                arr: Arr value consumed by this operation.
                v: V value consumed by this operation.
                mb_idx: Mb idx value consumed by this operation.
            """
            return arr[mb_idx] if V == 1 else arr[v, mb_idx]

        def si_set(arr, v, mb_idx, val):
            """Functional write of ``val`` into ``arr[v, mb_idx]`` (or ``arr[mb_idx]``).

            Args:
                arr: Arr value consumed by this operation.
                v: V value consumed by this operation.
                mb_idx: Mb idx value consumed by this operation.
                val: Val value consumed by this operation.
            """
            return arr.at[mb_idx].set(val) if V == 1 else arr.at[v, mb_idx].set(val)

        def g_add(acc, v, upd):
            """Accumulate ``upd`` into the grad accumulator at virt ``v``.

            Args:
                acc: Acc value consumed by this operation.
                v: V value consumed by this operation.
                upd: Upd value consumed by this operation.
            """
            if V == 1:
                return jax.tree.map(lambda a, b: a + b, acc, upd)
            return jax.tree.map(lambda a, b: a.at[v].add(b), acc, upd)

        for _t, row in enumerate(grid):
            for r, action in enumerate(row):
                if action is None:
                    continue
                mb = action.microbatch
                v = action.virtual_stage
                is_my = rank == r
                p_v = pick_virt(p_per_rank, v)
                logical = schedule.logical_at(r, v, n_stages)

                if action.phase == Phase.FWD:
                    x_in_source = xs[mb] if logical == 0 else si_get(incoming_fwd, v, mb)

                    def _fwd_branch(x_, _p=p_v):
                        """Active-rank forward: ``fwd_fn(params, x)``.

                        Args:
                            x_: X  value consumed by this operation.
                            _p:  p value consumed by this operation.
                        """
                        return fwd_fn(_p, x_)

                    def _fwd_skip(x_):
                        """Inactive-rank forward: produce zeros so all ranks have a valid out value.

                        Args:
                            x_: X  value consumed by this operation.
                        """
                        return jnp.zeros_like(x_)

                    y = jax.lax.cond(is_my, _fwd_branch, _fwd_skip, x_in_source)
                    saved_inputs = jnp.where(is_my, si_set(saved_inputs, v, mb, x_in_source), saved_inputs)
                    if logical == n_logical - 1:
                        saved_outputs = jnp.where(is_my, si_set(saved_outputs, v, mb, y), saved_outputs)
                    next_loc = schedule.next_logical_loc(r, v, n_stages)
                    if next_loc is not None:
                        dp, dv = next_loc
                        if dp == r:
                            incoming_fwd = jnp.where(is_my, si_set(incoming_fwd, dv, mb, y), incoming_fwd)
                        else:
                            y_sent = jax.lax.ppermute(y, pp_axis, perm=[(r, dp)])
                            incoming_fwd = jnp.where(rank == dp, si_set(incoming_fwd, dv, mb, y_sent), incoming_fwd)

                elif action.phase in (Phase.BWD, Phase.BWD_I, Phase.BWD_W):
                    if logical == n_logical - 1:

                        def _loss_branch(_so=saved_outputs, _v=v, _mb=mb):
                            """Active terminal-rank branch: pull saved output, return ``(loss, g_y)``.

                            Args:
                                _so:  so value consumed by this operation.
                                _v:  v value consumed by this operation.
                                _mb:  mb value consumed by this operation.
                            """
                            y_out = si_get(_so, _v, _mb)
                            loss_mb, g_y_mb = loss_and_g_y(y_out, *(t_arr[_mb] for t_arr in targets))
                            return loss_mb.astype(jnp.float32), g_y_mb

                        def _loss_skip():
                            """Inactive-rank branch: zero loss and zero seed cotangent."""
                            return jnp.zeros((), dtype=jnp.float32), jnp.zeros(mb_shape, dtype=mb_dtype)

                        loss_mb, g_y_seed = jax.lax.cond(is_my, _loss_branch, _loss_skip)
                        if action.phase != Phase.BWD_W:
                            loss_acc = loss_acc + loss_mb
                        g_y = g_y_seed
                    else:
                        g_y = si_get(incoming_bwd, v, mb)
                    x_saved = si_get(saved_inputs, v, mb)

                    def _bwd_branch(args, _p=p_v):
                        """Active-rank backward: ``bwd_fn(params, x_saved, g_y) -> (g_p, g_x)``.

                        Args:
                            args: Positional arguments forwarded to the wrapped callable.
                            _p:  p value consumed by this operation.
                        """
                        x_, g_ = args
                        return bwd_fn(_p, x_, g_)

                    def _bwd_skip(args, _p=p_v):
                        """Inactive-rank backward: zero param-grads and zero input-grad.

                        Args:
                            args: Positional arguments forwarded to the wrapped callable.
                            _p:  p value consumed by this operation.
                        """
                        x_, _g = args
                        return (jax.tree.map(jnp.zeros_like, _p), jnp.zeros_like(x_))

                    g_p_virt, g_x = jax.lax.cond(is_my, _bwd_branch, _bwd_skip, (x_saved, g_y))
                    if action.phase != Phase.BWD_I:
                        gated = jax.tree.map(lambda u, _m=is_my: jnp.where(_m, u, jnp.zeros_like(u)), g_p_virt)
                        g_params = g_add(g_params, v, gated)
                    prev_loc = _prev_logical_loc(schedule, r, v, n_stages)
                    if action.phase != Phase.BWD_W and prev_loc is not None:
                        sp, sv = prev_loc
                        if sp == r:
                            incoming_bwd = jnp.where(is_my, si_set(incoming_bwd, sv, mb, g_x), incoming_bwd)
                        else:
                            g_sent = jax.lax.ppermute(g_x, pp_axis, perm=[(r, sp)])
                            incoming_bwd = jnp.where(rank == sp, si_set(incoming_bwd, sv, mb, g_sent), incoming_bwd)

        total_loss = jax.lax.psum(loss_acc, pp_axis)
        mean_loss = total_loss / jnp.asarray(m, dtype=total_loss.dtype)
        return mean_loss, _add0(g_params)

    return body
