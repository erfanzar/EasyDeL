# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Schedule unit construction and dependency helpers for MPMD dispatch."""

from __future__ import annotations

from collections.abc import Callable

import jax

from ...core._weakcache import weak_invalidate
from ..schedules import FusedTask, Phase
from .schedule_types import _ApplyPayload, _ScheduleUnit

_SCHEDULE_FUSED_FWDBWD_CACHE: dict[tuple[int, ...], Callable[..., object]] = {}
_SCHEDULE_DIRECT_FUSED_FWDBWD_CACHE: dict[tuple[int, int], Callable[..., object]] = {}


def _get_schedule_fused_fwd_bwd_jit(
    fwd_jit: Callable[..., object],
    bwd_jit: Callable[..., object],
    n_invars: int,
) -> Callable[..., object]:
    """Return a cached jit that fuses FWD on microbatch A with BWD on microbatch B.

    The 1F1B steady state alternates a forward on a fresh microbatch
    with a backward on an in-flight one. Folding both into a single
    :func:`jax.jit` cuts the per-step Python dispatch count in half and
    lets XLA overlap the two computations on the same rank. The result
    is memoised in :data:`_SCHEDULE_FUSED_FWDBWD_CACHE` so subsequent
    fused units reuse the same compiled program.

    Args:
        fwd_jit: Per-cluster forward jit (from :func:`_make_fwd_jit`).
        bwd_jit: Per-cluster backward jit (from :func:`_make_bwd_jit`).
        n_invars: Number of cluster input variables (used to slice the
            packed ``(fwd_invars, bwd_invars, cotangents)`` tuple).

    Returns:
        Jitted ``(consts, *args) -> (fwd_outs, g_consts, g_bwd_invars)``
        callable.
    """
    key = (id(fwd_jit), id(bwd_jit), n_invars)
    cached = _SCHEDULE_FUSED_FWDBWD_CACHE.get(key)
    if cached is not None:
        return cached

    @jax.jit
    def fused(consts: tuple[object, ...], *args: object) -> tuple[object, ...]:
        """One-launch fwd(A) + bwd(B) for paired schedule microbatches.

        ``args`` is the concatenation of ``(fwd_invars, bwd_invars,
        cotangents)``, each of length ``n_invars`` (cotangents may be
        a different length depending on the cluster's outputs). The
        function returns ``(fwd_outs, g_consts, g_bwd_invars)`` so the
        caller can route the forward outputs to the next stage and
        accumulate the backward gradients.

        Args:
            consts: Consts value consumed by this operation.
            *args: Additional positional arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        with jax.named_scope("spectrax/mpmd/schedule/fused_fwdbwd"):
            fwd_invars = args[:n_invars]
            bwd_invars = args[n_invars : 2 * n_invars]
            cotangents = args[2 * n_invars :]
            with jax.named_scope("spectrax/mpmd/schedule/fused_fwdbwd/forward"):
                fwd_outs = fwd_jit(consts, *fwd_invars)
            with jax.named_scope("spectrax/mpmd/schedule/fused_fwdbwd/backward"):
                g_consts, g_bwd_invars = bwd_jit(consts, *bwd_invars, *cotangents)
            return fwd_outs, g_consts, g_bwd_invars

    _SCHEDULE_FUSED_FWDBWD_CACHE[key] = fused
    weak_invalidate(fwd_jit, _SCHEDULE_FUSED_FWDBWD_CACHE, key)
    weak_invalidate(bwd_jit, _SCHEDULE_FUSED_FWDBWD_CACHE, key)
    return fused


def _eval_schedule_cluster_fwd(cluster_jaxpr: object, consts: tuple[object, ...], *invars: object) -> tuple[object, ...]:
    """Evaluate a cluster sub-jaxpr without nesting a pre-compiled stage jit.

    Used by direct-fused dispatch paths that compose multiple cluster
    operations inside a single :func:`jax.jit`; calling the precompiled
    forward jit instead would force a re-trace on every invocation.

    Args:
        cluster_jaxpr: The stage's sub-jaxpr.
        consts: Placed constants for that cluster.
        *invars: Stage input activations.

    Returns:
        Tuple of stage outputs (matching the cluster's outvars).
    """
    return tuple(jax.core.eval_jaxpr(cluster_jaxpr, list(consts), *invars))


def _eval_schedule_cluster_bwd(
    cluster_jaxpr: object,
    n_invars: int,
    consts: tuple[object, ...],
    *invars_and_cotangents: object,
) -> tuple[object, tuple[object, ...]]:
    """Compute ``(g_consts, g_invars)`` for a cluster inside a fused jit.

    Mirrors :func:`_make_bwd_jit` but calls the cluster jaxpr inline so
    a fused fwd+bwd jit can carry both halves in one HLO graph.

    Args:
        cluster_jaxpr: The stage's sub-jaxpr.
        n_invars: Number of cluster invars (used to slice
            ``invars_and_cotangents``).
        consts: Placed constants for the cluster.
        *invars_and_cotangents: ``(*invars, *cotangents)`` packed as a
            single positional sequence.

    Returns:
        ``(g_consts, g_invars)`` aligned with ``consts`` and the
        cluster's invars respectively.
    """
    invars = invars_and_cotangents[:n_invars]
    cotangents = invars_and_cotangents[n_invars:]

    def pure(c: tuple[object, ...], *xs: object) -> tuple[object, ...]:
        """Pure (consts, invars) -> outs interpreter for ``jax.vjp`` linearization.

        Args:
            c: C value consumed by this operation.
            *xs: Additional positional arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        return tuple(jax.core.eval_jaxpr(cluster_jaxpr, list(c), *xs))

    _, vjp_fn = jax.vjp(pure, consts, *invars)
    grads = vjp_fn(tuple(cotangents))
    return grads[0], tuple(grads[1:])


def _eval_schedule_cluster_terminal(
    cluster_jaxpr: object,
    n_invars: int,
    consts: tuple[object, ...],
    *invars: object,
) -> tuple[object, tuple[object, tuple[object, ...]]]:
    """Run the terminal cluster's loss + gradient computation in-place.

    Used as a building block for direct-fused jits that compose the
    cluster jaxpr inline rather than calling a pre-compiled stage jit.
    Wraps the cluster's scalar-loss evaluator in
    :func:`jax.value_and_grad` over both ``consts`` and ``invars`` so a
    single trace produces the loss value and the seed cotangents the
    upstream backward sweep needs.

    Args:
        cluster_jaxpr: Terminal cluster sub-jaxpr.
        n_invars: Cluster's invar count.
        consts: Placed cluster constants.
        *invars: Cluster's positional inputs.

    Returns:
        ``(loss, (g_consts, g_invars))`` mirroring
        :func:`_make_terminal_jit`'s output.
    """

    def pure(c: tuple[object, ...], *xs: object) -> object:
        """Pure (consts, invars) -> scalar interpreter for the loss cluster.

        The terminal cluster is required to produce a single scalar
        (the per-microbatch loss). Anything else is a tracing error.

        Args:
            c: C value consumed by this operation.
            *xs: Additional positional arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        outs = jax.core.eval_jaxpr(cluster_jaxpr, list(c), *xs)
        if len(outs) != 1:
            raise ValueError(
                f"Terminal cluster must produce exactly one scalar output (the per-microbatch loss); got {len(outs)}."
            )
        return outs[0]

    argnums = tuple(range(1 + n_invars))
    loss, grads = jax.value_and_grad(pure, argnums=argnums, allow_int=True)(consts, *invars)
    return loss, (grads[0], tuple(grads[1:]))


def _get_schedule_direct_fused_fwd_bwd_jit(
    cluster_jaxpr: object,
    n_invars: int,
) -> Callable[..., object]:
    """Return a cached fused FWD(A)+BWD(B) jit built directly from the cluster jaxpr.

    Variant of :func:`_get_schedule_fused_fwd_bwd_jit` that bypasses
    the per-cluster pre-compiled forward/backward jits and re-evaluates
    the cluster's jaxpr inside one outer :func:`jax.jit`. This lets
    XLA see the entire fwd+bwd as a single graph (better fusion) at
    the cost of recompiling the cluster body inside this jit.

    Cached on ``(id(cluster_jaxpr), n_invars)`` in
    :data:`_SCHEDULE_DIRECT_FUSED_FWDBWD_CACHE`.

    Args:
        cluster_jaxpr: The stage's sub-jaxpr.
        n_invars: Number of cluster input variables.

    Returns:
        Jitted ``(consts, *args) -> (fwd_outs, g_consts, g_bwd_invars)``
        callable.
    """
    key = (id(cluster_jaxpr), n_invars)
    cached = _SCHEDULE_DIRECT_FUSED_FWDBWD_CACHE.get(key)
    if cached is not None:
        return cached

    @jax.jit
    def fused(consts: tuple[object, ...], *args: object) -> tuple[object, ...]:
        """Fused fwd(A)+bwd(B) that evaluates the cluster jaxpr directly.

        Bypasses the per-stage compiled fwd/bwd JITs (used elsewhere)
        and instead lets XLA compile the entire fwd+bwd as a single
        function. Profile-driven cache: the result is keyed on
        ``(id(cluster_jaxpr), n_invars)``.

        Args:
            consts: Consts value consumed by this operation.
            *args: Additional positional arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        with jax.named_scope("spectrax/mpmd/schedule/direct_fused_fwdbwd"):
            fwd_invars = args[:n_invars]
            bwd_invars = args[n_invars : 2 * n_invars]
            cotangents = args[2 * n_invars :]
            with jax.named_scope("spectrax/mpmd/schedule/direct_fused_fwdbwd/forward"):
                fwd_outs = _eval_schedule_cluster_fwd(cluster_jaxpr, consts, *fwd_invars)
            with jax.named_scope("spectrax/mpmd/schedule/direct_fused_fwdbwd/backward"):
                g_consts, g_bwd_invars = _eval_schedule_cluster_bwd(
                    cluster_jaxpr,
                    n_invars,
                    consts,
                    *bwd_invars,
                    *cotangents,
                )
            return fwd_outs, g_consts, g_bwd_invars

    _SCHEDULE_DIRECT_FUSED_FWDBWD_CACHE[key] = fused
    return fused


def _schedule_action_unit(
    *,
    index: int,
    row: int,
    rank: int,
    action: object,
    logical_for_loc: dict[tuple[int, int], int],
    logical_override: int | None = None,
) -> _ScheduleUnit:
    """Wrap a single :class:`Action` cell as a :class:`_ScheduleUnit`.

    Forward actions populate ``fwd_logical``/``fwd_mb``; backward
    actions populate ``bwd_logical``/``bwd_mb``/``bwd_phase``. Used by
    :func:`_build_schedule_units_from_plan` when expanding the
    schedule grid into the dependency DAG.

    Args:
        index: Stable global ordering key.
        row: Source row in the schedule grid.
        rank: Physical pipeline rank.
        action: The :class:`Action` cell.
        logical_for_loc: Mapping from ``(rank, virt)`` to logical
            stage index.

    Returns:
        A :class:`_ScheduleUnit` reflecting the action's phase.
    """
    virt = action.virtual_stage
    logical = logical_for_loc[(rank, virt)] if logical_override is None else logical_override
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


def _schedule_fused_unit(
    *,
    index: int,
    row: int,
    rank: int,
    fused: FusedTask,
    logical_for_loc: dict[tuple[int, int], int],
    logical_override: int | None = None,
    fwd_logical_override: int | None = None,
    bwd_logical_override: int | None = None,
) -> _ScheduleUnit:
    """Wrap a paired FWD+BWD :class:`FusedTask` as a single :class:`_ScheduleUnit`.

    The two halves always share the same physical rank, but DualPipe-V
    can pair different virtual stages on that rank. Carry separate
    logical-stage indices so the dispatcher routes each half through
    the correct stage state.

    Args:
        index: Stable global ordering key.
        row: Source row in the schedule grid.
        rank: Physical pipeline rank.
        fused: The :class:`FusedTask` cell.
        logical_for_loc: Mapping from ``(rank, virt)`` to logical
            stage index.

    Returns:
        A :class:`_ScheduleUnit` with ``kind="fused"``.
    """
    virt = fused.virtual_stage
    if logical_override is not None:
        fwd_logical = logical_override
        bwd_logical = logical_override
    else:
        fwd_logical = (
            logical_for_loc[(rank, fused.fwd.virtual_stage)] if fwd_logical_override is None else fwd_logical_override
        )
        bwd_logical = (
            logical_for_loc[(rank, fused.bwd.virtual_stage)] if bwd_logical_override is None else bwd_logical_override
        )
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


def _schedule_apply_unit(
    *,
    index: int,
    row: int,
    rank: int,
    virt: int = 0,
) -> _ScheduleUnit:
    """Wrap a per-rank optimizer-apply step as a :class:`_ScheduleUnit`.

    APPLY units are not present in the schedule grid -- they are
    synthesized at unit-build time by ``_build_schedule_units_from_plan``
    when the plan has compiled apply jits. Each unit owns one rank's
    stage-local parameter update, and depends on every backward unit on
    its rank (the dependency edges are added in
    :func:`_build_schedule_unit_dependencies`). The ``row`` slot is set
    to a synthetic value past every real grid row so deterministic
    dispatch fires the apply units after all fwd/bwd work on the same
    rank has been issued.

    Args:
        index: Stable global ordering key.
        row: Synthetic row index (past every real grid row).
        rank: Physical pipeline rank whose stage-local leaves this unit
            will update.
        virt: Virtual sub-stage index (unused for apply; kept for
            symmetry with fwd/bwd units).

    Returns:
        A :class:`_ScheduleUnit` with ``kind="apply"`` and an
        :class:`_ApplyPayload` payload.
    """
    return _ScheduleUnit(
        index=index,
        row=row,
        kind="apply",
        rank=rank,
        virt=virt,
        payload=_ApplyPayload(rank=rank, virt=virt),
        fwd_logical=None,
        fwd_mb=None,
        bwd_logical=None,
        bwd_mb=None,
        bwd_phase=None,
    )


def _append_apply_units(
    plan: dict[str, object],
    units: list[_ScheduleUnit],
    next_index: int,
) -> int:
    """Append one APPLY unit per physical rank when the plan has apply jits.

    No-op when the plan does not contain compiled ``apply_jits`` -- this
    keeps the legacy ``sxvalue_and_grad`` path (no fused apply) byte-for-
    byte unchanged. When ``apply_jits`` is present, emits exactly one
    APPLY unit per ``rank`` (``virt=0``), placed after every real grid
    unit in row-major order so deterministic dispatch issues them last
    on each rank but other ranks can fire theirs in parallel with rank-0
    bwd tail.

    Args:
        plan: Dispatch plan from :func:`_build_schedule_plan`.
        units: Unit list to append onto.
        next_index: Next stable global ordering key to assign.

    Returns:
        The updated ``next_index`` after appending.
    """
    apply_jits = plan.get("apply_jits")
    if not apply_jits:
        return next_index
    n_rank = int(plan.get("n", 0))
    if n_rank <= 0:
        return n_rank
    grid = plan.get("grid", ())
    synthetic_row = len(grid) + 1
    for rank in range(n_rank):
        if (rank, 0) not in apply_jits:
            continue
        units.append(_schedule_apply_unit(index=next_index, row=synthetic_row, rank=rank, virt=0))
        next_index += 1
    return next_index


def _fuse_cross_virtual_schedule_units(plan: dict[str, object], units: list[_ScheduleUnit]) -> list[_ScheduleUnit]:
    """Return schedule units without hidden env-gated fusion.

    Real schedule cells such as ``FusedTask`` are still preserved by
    ``_build_schedule_units_from_plan``. This helper intentionally does not
    rewrite adjacent rows through a private OS flag; that keeps the MPMD
    runtime's behavior explicit and benchmarkable.

    Args:
        plan: Plan value consumed by this operation.
        units: Units value consumed by this operation.

    Returns:
        Return schedule units without hidden env-gated fusion.
    """
    del plan
    return units


def _build_schedule_units_from_plan(plan: dict[str, object]) -> list[_ScheduleUnit]:
    """Walk the schedule grid and emit executable units in row-major order.

    Schedules that want split backward must explicitly emit ``BWD_I`` and
    ``BWD_W`` actions. A plain ``BWD`` action is kept as one full backward
    unit; automatically splitting it here duplicates VJP work and changes the
    scheduler's intended critical path. The terminal-rank backward action is
    omitted when ``eager_terminal_bwd`` is set (the terminal backward is fired
    eagerly inside the forward stub).

    Args:
        plan: Dispatch plan from :func:`_build_schedule_plan`.

    Returns:
        Ordered list of :class:`_ScheduleUnit` objects ready for
        dependency analysis.
    """
    units: list[_ScheduleUnit] = []
    next_index = 0
    logical_for_loc = plan["logical_for_loc"]
    n_logical = plan["n_logical"]
    schedule_n_logical = plan.get("schedule_n_logical", n_logical)
    serial_region_plan = bool(plan.get("serial_region_plan", False))
    region_groups = (n_logical // schedule_n_logical) if serial_region_plan else 1
    terminal_logical = plan.get("terminal_logical", n_logical - 1)
    eager_terminal_bwd = False
    grid = plan["grid"]

    def _is_eager_terminal_bwd(action: object, logical: int) -> bool:
        return (
            eager_terminal_bwd and logical == terminal_logical and action.phase in (Phase.BWD, Phase.BWD_I, Phase.BWD_W)
        )

    if serial_region_plan:

        def emit_action(
            *,
            logical: int,
            synthetic_row: int,
            rank: int,
            action: object,
        ) -> None:
            nonlocal next_index
            if _is_eager_terminal_bwd(action, logical):
                return
            units.append(
                _schedule_action_unit(
                    index=next_index,
                    row=synthetic_row,
                    rank=rank,
                    action=action,
                    logical_for_loc=logical_for_loc,
                    logical_override=logical,
                )
            )
            next_index += 1

        def append_action(
            *,
            logical: int,
            synthetic_row: int,
            rank: int,
            action: object,
        ) -> None:
            """Append one scheduler action without rewriting its phase."""
            emit_action(logical=logical, synthetic_row=synthetic_row, rank=rank, action=action)

        for group in range(region_groups):
            logical_offset = group * schedule_n_logical
            row_offset = group * len(grid)
            for row_idx, row in enumerate(grid):
                for rank, cell in enumerate(row):
                    if cell is None:
                        continue
                    fwd_action = cell.fwd if isinstance(cell, FusedTask) else cell
                    if fwd_action.phase is not Phase.FWD:
                        continue
                    logical = logical_offset + logical_for_loc[(rank, fwd_action.virtual_stage)]
                    if logical < n_logical:
                        append_action(logical=logical, synthetic_row=row_offset + row_idx, rank=rank, action=fwd_action)

        bwd_row_base = region_groups * len(grid)
        for reverse_group, group in enumerate(reversed(range(region_groups))):
            logical_offset = group * schedule_n_logical
            row_offset = bwd_row_base + reverse_group * len(grid)
            for row_idx, row in enumerate(grid):
                for rank, cell in enumerate(row):
                    if cell is None:
                        continue
                    bwd_actions = (cell.fwd, cell.bwd) if isinstance(cell, FusedTask) else (cell,)
                    for bwd_action in bwd_actions:
                        if bwd_action.phase not in (Phase.BWD, Phase.BWD_I, Phase.BWD_W):
                            continue
                        logical = logical_offset + logical_for_loc[(rank, bwd_action.virtual_stage)]
                        if logical < n_logical:
                            append_action(
                                logical=logical,
                                synthetic_row=row_offset + row_idx,
                                rank=rank,
                                action=bwd_action,
                            )
        next_index = _append_apply_units(plan, units, next_index)
        return _fuse_cross_virtual_schedule_units(plan, units)

    for group in range(region_groups):
        logical_offset = group * schedule_n_logical
        row_offset = group * len(grid)
        for row_idx, row in enumerate(grid):
            for rank, cell in enumerate(row):
                if cell is None:
                    continue
                base_logical = logical_for_loc[(rank, cell.virtual_stage)]
                logical = logical_offset + base_logical
                if logical >= n_logical:
                    continue
                synthetic_row = row_offset + row_idx
                if isinstance(cell, FusedTask):
                    fwd_logical = logical_offset + logical_for_loc[(rank, cell.fwd.virtual_stage)]
                    bwd_logical = logical_offset + logical_for_loc[(rank, cell.bwd.virtual_stage)]
                    if (
                        cell.fwd.phase is Phase.FWD
                        and cell.bwd.phase is Phase.BWD
                        and fwd_logical < n_logical
                        and bwd_logical < n_logical
                        and not _is_eager_terminal_bwd(cell.bwd, bwd_logical)
                    ):
                        units.append(
                            _schedule_fused_unit(
                                index=next_index,
                                row=synthetic_row,
                                rank=rank,
                                fused=cell,
                                logical_for_loc=logical_for_loc,
                                fwd_logical_override=fwd_logical,
                                bwd_logical_override=bwd_logical,
                            )
                        )
                        next_index += 1
                        continue

                    actions = (cell.fwd, cell.bwd)
                    for action in actions:
                        action_logical = logical_offset + logical_for_loc[(rank, action.virtual_stage)]
                        if action_logical >= n_logical:
                            continue
                        if _is_eager_terminal_bwd(action, action_logical):
                            continue
                        units.append(
                            _schedule_action_unit(
                                index=next_index,
                                row=synthetic_row,
                                rank=rank,
                                action=action,
                                logical_for_loc=logical_for_loc,
                                logical_override=action_logical,
                            )
                        )
                        next_index += 1
                else:
                    if _is_eager_terminal_bwd(cell, logical):
                        continue
                    units.append(
                        _schedule_action_unit(
                            index=next_index,
                            row=synthetic_row,
                            rank=rank,
                            action=cell,
                            logical_for_loc=logical_for_loc,
                            logical_override=logical,
                        )
                    )
                    next_index += 1
    next_index = _append_apply_units(plan, units, next_index)
    return _fuse_cross_virtual_schedule_units(plan, units)


def _build_schedule_unit_dependencies(plan: dict[str, object], units: list[_ScheduleUnit]) -> dict[int, set[int]]:
    """Compute the predecessor-set for each schedule unit.

    Three classes of edge are added:

    * **Same-rank order**: each unit depends on the previous unit fired
      on the same rank (preserves the schedule's intended sequencing).
    * **Forward-data dependencies**: a forward unit depends on the
      forward units that produced each of its cluster inputs (looked
      up via ``invar_sources``).
    * **Backward-cotangent dependencies**: a backward unit depends on
      its own paired forward (so saved activations are available) and
      on the backward of every downstream consumer that supplies a
      cotangent. ``BWD_W`` units are excluded from
      cotangent-supplier tracking because they only produce weight
      grads.

    Args:
        plan: Dispatch plan from :func:`_build_schedule_plan`.
        units: Units returned by :func:`_build_schedule_units_from_plan`.

    Returns:
        A mapping ``unit_index -> {predecessor unit indices}``.
    """
    n_logical = plan["n_logical"]
    invar_sources = plan["invar_sources"]
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

    terminal_logical = plan.get("terminal_logical", n_logical - 1)
    for mb in range(plan["m"]):
        fwd_idx = fwd_units.get((terminal_logical, mb))
        if fwd_idx is not None and (terminal_logical, mb) not in bwd_cot_units:
            bwd_cot_units[(terminal_logical, mb)] = fwd_idx

    deps: dict[int, set[int]] = {unit.index: set() for unit in units}
    previous_by_rank: dict[int, int] = {}

    def add_dep(unit: _ScheduleUnit, dep: int | None) -> None:
        """Insert ``dep`` into ``unit``'s predecessor set if it is real and distinct.

        Skips ``None`` (no dependency) and self-references (a unit
        cannot depend on itself). The caller may pass a missing
        index from a dictionary lookup directly without an extra
        ``if`` check.

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

    if bool(plan.get("apply_requires_all_grads", False)):
        all_bwd_units = tuple(unit.index for unit in units if unit.bwd_logical is not None and unit.bwd_mb is not None)
        for unit in units:
            if unit.kind == "apply":
                deps[unit.index].update(dep for dep in all_bwd_units if dep != unit.index)

    return deps


def _dependency_topological_schedule_units(
    units: list[_ScheduleUnit],
    deps: dict[int, set[int]],
) -> list[_ScheduleUnit]:
    """Return units in a stable dependency-compatible order.

    The order is used for deterministic multi-controller transport gates. It
    must be topologically compatible with the schedule DAG; a plain row-major
    walk can ask the gate to wait for a transfer whose owning unit is not
    dependency-ready yet.
    """
    by_index = {unit.index: unit for unit in units}
    dependents: dict[int, set[int]] = {unit.index: set() for unit in units}
    remaining = {idx: set(unit_deps) for idx, unit_deps in deps.items()}
    for idx, unit_deps in deps.items():
        for dep in unit_deps:
            dependents.setdefault(dep, set()).add(idx)

    ready = sorted(
        (idx for idx, unit_deps in remaining.items() if not unit_deps),
        key=lambda idx: (by_index[idx].row, idx),
    )
    ordered: list[_ScheduleUnit] = []
    emitted: set[int] = set()
    while ready:
        idx = ready.pop(0)
        if idx in emitted:
            continue
        emitted.add(idx)
        ordered.append(by_index[idx])
        for dependent in dependents.get(idx, ()):
            remaining[dependent].discard(idx)
            if not remaining[dependent] and dependent not in emitted and dependent not in ready:
                ready.append(dependent)
        ready.sort(key=lambda item: (by_index[item].row, item))

    if len(ordered) != len(units):
        blocked = {idx: sorted(unit_deps) for idx, unit_deps in remaining.items() if idx not in emitted and unit_deps}
        raise RuntimeError(f"schedule executor dependency cycle or missing dependency: {blocked}")
    return ordered
