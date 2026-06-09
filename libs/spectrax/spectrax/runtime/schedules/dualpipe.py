# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DualPipe-V schedule and per-rank task builder (DeepSeek-V3).

DualPipe-V is the V-shaped bidirectional schedule used by DeepSeek-V3.
Every physical rank hosts two virtual stages: one that runs in the
forward direction (rank ``r`` -> ``r+1``) and one that runs in the
reverse direction (rank ``r+1`` -> ``r``). The two virtual stages
mirror each other so each microbatch visits every rank twice before
hitting the loss.

This module provides:

* :class:`DualPipeV` — the :class:`Schedule` subclass that emits the
  V-shape grid for any spectrax pipeline runtime.
* :func:`dualpipev_tasks` — a per-rank, eight-section task list that
  matches DeepSeek's reference ``dualpipev.py``. Useful for building
  custom executors or experimental runners outside spectrax's grid
  representation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import Action, FusedTask, Phase, Schedule


@dataclass
class DualPipeV(Schedule):
    """DualPipe-V: V-shaped bidirectional pipeline (DeepSeek).

    Every physical rank hosts **two** virtual stages in a V topology:
    rank ``r`` owns logical stage ``r`` (forward direction) and
    logical stage ``2n - 1 - r`` (reverse direction). Activations flow
    through physical ranks ``0 -> n-1``, then bounce back ``n-1 -> 0``,
    so each microbatch visits every rank twice before the loss.

    Pros:

    * Halves the pipeline bubble vs :class:`Std1F1B` at the same peak
      activation memory (the mirrored stage fills what would be idle).
    * End-to-end latency comparable to :class:`InterleavedH1` but
      without the cross-rank ppermute-per-virtual-stage cost — the
      second virtual stage is adjacent to the first on the same rank.

    Cons:

    * Requires ``n_stages`` pipeline ranks for ``2 * n_stages``
      logical stages — callers must structure the model as
      ``2 * n_stages`` :class:`PipelineSequential` entries.
    * Requires runtime support for mixed-virtual FWD+BWD cells to
      overlap the paired work; otherwise those cells are serialized
      and the schedule loses much of its intended benefit.

    Reference: DeepSeek-V3 technical report; DeepSeek
    ``dualpipev.py``.

    Attributes:
        microbatches: see :class:`Schedule`.
        zero_bubble: When ``True``, emit the DeepSeek split
            ``BWD_I``/``BWD_W`` slots. Set ``False`` to keep the same
            V-shaped FWD/BWD pairing while using full backward chunks
            for runtimes where split VJPs are slower than bubble fill.
    """

    zero_bubble: bool = True

    def build(self, n_stages: int) -> list[list[Action | FusedTask | None]]:
        """Emit the DeepSeek-style per-rank DualPipe-V task grid.

        The reference DualPipe-V algorithm is rank-centric rather than
        global-time-step-centric: each physical rank runs an eight-section
        task list containing warmup forwards, split ``BWD_I``/``BWD_W``
        work, and forward/backward overlap points.  Build that per-rank
        sequence with :func:`dualpipev_tasks`, then pad the ragged lists
        into SpectraX's grid representation.  The MPMD runtime preserves
        same-rank order and adds data dependencies between ranks.

        Args:
            n_stages: N stages value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        n = n_stages
        m = self.microbatches
        if m < 2 * n:
            raise ValueError(
                "DualPipeV requires microbatches >= 2 * n_stages to match the reference schedule; "
                f"got microbatches={m}, n_stages={n}."
            )

        per_rank_tasks = [dualpipev_tasks(n, rank, m, zero_bubble=self.zero_bubble) for rank in range(n)]
        max_len = max((len(tasks) for tasks in per_rank_tasks), default=0)
        grid: list[list[Action | FusedTask | None]] = []
        for row_idx in range(max_len):
            grid.append([tasks[row_idx] if row_idx < len(tasks) else None for tasks in per_rank_tasks])
        return grid

    def virtual_stages_per_rank(self) -> int:
        """Always ``2`` for the V-shape topology (forward and reverse virtuals).

        Returns:
            Result described by this helper.
        """
        return 2

    def logical_at(self, rank: int, virt: int, n_stages: int) -> int:
        """Map ``(rank, virt)`` to its logical stage under the V shape.

        ``virt == 0`` traces ranks ``0..n-1`` as logical stages
        ``0..n-1`` (forward direction); ``virt == 1`` traces ranks
        ``n-1..0`` as logical stages ``n..2n-1`` (reverse direction).

        Args:
            rank: Physical rank index in ``[0, n_stages)``.
            virt: Virtual-stage index, ``0`` (forward leg) or ``1``
                (reverse leg).
            n_stages: Number of physical pipeline ranks.

        Returns:
            Logical-stage index in ``[0, 2 * n_stages)``.
        """
        return rank if virt == 0 else 2 * n_stages - 1 - rank

    def next_logical_loc(self, rank: int, virt: int, n_stages: int):
        """Return the ``(rank, virt)`` of the next logical stage, or ``None``.

        Walks the V-shape: in the forward leg the next stage is the
        next physical rank; at the apex (logical ``n - 1``) the path
        bounces and starts walking ranks back down on virtual stage 1.

        Args:
            rank: Current physical rank.
            virt: Current virtual-stage index.
            n_stages: Number of physical pipeline ranks.

        Returns:
            Downstream ``(rank, virt)`` or ``None`` if this is the
            terminal logical stage (``logical == 2 * n_stages - 1``).
        """
        current = self.logical_at(rank, virt, n_stages)
        nxt = current + 1
        if nxt >= 2 * n_stages:
            return None
        if nxt < n_stages:
            return (nxt, 0)
        return (2 * n_stages - 1 - nxt, 1)

    def terminal_loc(self, n_stages: int) -> tuple[int, int]:
        """Return ``(0, 1)`` — terminal logical stage ``2n-1`` lives on rank 0's reverse leg.

        Because the V bounces back, the final logical stage runs on
        physical rank ``0``'s reverse virtual stage. The runtime fires
        ``loss_fn`` here.

        Args:
            n_stages: Number of physical pipeline ranks (unused).

        Returns:
            Always ``(0, 1)``.
        """
        return (0, 1)

    def peak_activations(self, n_stages: int) -> int:
        """Peak ≈ ``2 * n_stages`` saved activations per rank (one per virtual leg).

        Each rank holds ``n_stages`` saved activations from the
        forward leg and another ``n_stages`` from the reverse leg.

        Args:
            n_stages: Number of physical pipeline ranks.

        Returns:
            ``2 * n_stages``.
        """
        return 2 * n_stages


def dualpipev_tasks(
    mpmd_dim: int,
    mpmd_idx: int,
    n_mubatches: int,
    *,
    zero_bubble: bool = True,
) -> list[Action | FusedTask]:
    """Per-rank task list for DualPipe-V (DeepSeek-V3).

    Returns the ordered task sequence a single physical rank
    (``mpmd_idx``) executes under DualPipe-V for ``n_mubatches``.
    Each task is either a plain :class:`Action` (pure FWD or pure
    BWD_I/BWD_W on one virtual stage) or a :class:`FusedTask` pairing
    a forward on one virtual stage with a (split) backward on the
    other — the steady-state workhorse that DeepSeek's kernel uses
    to overlap compute across the V.

    The construction is the 8-section rank-centric formulation from
    Based on the DeepSeek DualPipe-V reference implementation.
    Sections::

        1. nF0                     — warmup fwd on stage0 only
        2. nF0F1                   — warmup alternating fwd stage0 / stage1
        3. nB1W1F1                 — zero-bubble (bwd_i, bwd_w, fwd) triplet
        4. nF0B1F1B0 (main step)   — steady state, fused fwd/bwd pairs
        5. nB1F1B0                 — cooldown start
        6. nB1B0                   — mid cooldown; enable ZB for some ranks
        7. nWB0                    — ZB-only cooldown
        8. nW                      — final weight-grad flushes

    Each rank owns two virtual stages: ``stage0 = mpmd_idx`` and
    ``stage1 = 2 * mpmd_dim - 1 - mpmd_idx``. The backward direction
    traverses ``stage1 -> stage0`` per microbatch.

    Args:
        mpmd_dim: Number of physical pipeline ranks.
        mpmd_idx: Index of the rank whose task list to build.
        n_mubatches: Number of microbatches per step.
        zero_bubble: Emit split BWD-I/BWD-W chunks for the zero-bubble
            sections. When false, those chunks become ordinary full BWD
            tasks and W-flush slots are omitted.

    Returns:
        A list of :class:`Action` / :class:`FusedTask` entries in
        execution order for ``mpmd_idx``. Callers that want a
        time-step grid across all ranks (spectrax's standard
        representation) should use :class:`DualPipeV`.\\ :meth:`build`
        instead — this function is for building custom per-rank
        executors or experimental schedule runners.
    """
    fwd_counts: dict[int, int] = {}
    bwd_counts: dict[int, int] = {}
    pending_w: list[tuple[int, int]] = []

    def _virt(stage_id: int) -> int:
        return 0 if stage_id < mpmd_dim else 1

    def _next_fwd_mb(stage_id: int) -> int:
        """Return the next forward microbatch index for ``stage_id``."""
        mb = fwd_counts.get(stage_id, 0)
        fwd_counts[stage_id] = mb + 1
        return mb

    def _next_bwd_mb(stage_id: int) -> int:
        """Return the next backward microbatch index for ``stage_id``.

        Full ``BWD`` and split ``BWD_I`` share this stream: DeepSeek's
        ``_backward_chunk`` always consumes exactly one backward
        microbatch, with ``enable_zb`` deciding whether weight-gradient
        work runs now or is queued for a later ``W`` slot.
        """
        mb = bwd_counts.get(stage_id, 0)
        bwd_counts[stage_id] = mb + 1
        return mb

    def _bwd_w_action(stage_id: int, mb: int) -> Action:
        return Action(Phase.BWD_W, mb, _virt(stage_id))

    def fwd(stage_id: int) -> Action:
        """Build a forward :class:`Action` for ``stage_id`` at its next microbatch.

        Virtual stage 0 for stages ``< mpmd_dim`` (the forward leg of
        the V), virtual stage 1 for the rest (the reverse leg) —
        matching the DualPipe-V layout where each rank owns two
        virtual stages.

        Args:
            stage_id: Logical-stage index.

        Returns:
            A FWD :class:`Action` with the right virtual-stage tag.
        """
        return Action(Phase.FWD, _next_fwd_mb(stage_id), _virt(stage_id))

    def bwd(stage_id: int, *, enable_zb: bool = False) -> Action:
        """Build one DeepSeek backward chunk.

        DeepSeek only splits backward when ``enable_zb`` is set. Normal
        backward chunks compute input and weight gradients together as
        ``BWD``. Zero-bubble chunks compute ``BWD_I`` now and queue the
        matching ``BWD_W`` for a later ``weight_chunk`` slot.
        """
        mb = _next_bwd_mb(stage_id)
        if zero_bubble and enable_zb:
            pending_w.append((stage_id, mb))
            return Action(Phase.BWD_I, mb, _virt(stage_id))
        return Action(Phase.BWD, mb, _virt(stage_id))

    def weight_chunk() -> Action:
        """Pop the next queued zero-bubble W-grad task."""
        if not pending_w:
            raise RuntimeError("DualPipeV internal schedule error: W slot had no queued BWD_W task.")
        stage_id, mb = pending_w.pop(0)
        return _bwd_w_action(stage_id, mb)

    def fwd_bwd(fwd_stage: int, bwd_stage: int) -> FusedTask:
        """Build a paired (FWD on ``fwd_stage``, full BWD on ``bwd_stage``) task.

        In the DeepSeek loop, ``_forward_backward_chunk`` uses the
        module's custom ``overlapped_forward_backward`` implementation
        when available; otherwise it executes a forward followed by a
        normal full backward. It is not a zero-bubble W-grad deferral
        point.

        Args:
            fwd_stage: Logical stage to forward on.
            bwd_stage: Logical stage to backward on (typically the
                mirror partner of ``fwd_stage``).

        Returns:
            A :class:`FusedTask` whose ``fwd`` is the forward action
            and whose ``bwd`` is the full backward action.
        """
        f = fwd(fwd_stage)
        b = bwd(bwd_stage)
        return FusedTask(fwd=f, bwd=b, virtual_stage=f.virtual_stage)

    stage0 = mpmd_idx
    stage1 = mpmd_dim * 2 - mpmd_idx - 1
    tasks: list[Action | FusedTask] = []

    section_1 = (mpmd_dim - mpmd_idx - 1) * 2
    tasks.extend(fwd(stage0) for _ in range(section_1))

    section_2 = mpmd_idx + 1
    for _ in range(section_2):
        tasks.append(fwd(stage0))
        tasks.append(fwd(stage1))

    section_3 = mpmd_dim - mpmd_idx - 1
    for _ in range(section_3):
        tasks.append(bwd(stage1, enable_zb=True))
        if zero_bubble:
            tasks.append(weight_chunk())
        tasks.append(fwd(stage1))

    section_4 = n_mubatches - mpmd_dim * 2 + mpmd_idx + 1
    for idx in range(section_4):
        if idx == 0:
            if mpmd_idx == mpmd_dim - 1:
                tasks.append(fwd(stage0))
                tasks.append(bwd(stage1))
            else:
                tasks.append(fwd_bwd(stage0, stage1))
        else:
            tasks.append(fwd_bwd(stage0, stage1))
        tasks.append(fwd_bwd(stage1, stage0))

    section_5 = mpmd_dim - mpmd_idx - 1
    for _ in range(section_5):
        tasks.append(bwd(stage1))
        tasks.append(fwd_bwd(stage1, stage0))

    section_6 = mpmd_idx + 1
    enable_zb_at = section_6 // 2
    enable_zb = False
    for idx in range(section_6):
        if idx == enable_zb_at and mpmd_idx % 2 == 1:
            enable_zb = True
        tasks.append(bwd(stage1, enable_zb=enable_zb))
        if idx == enable_zb_at and mpmd_idx % 2 == 0:
            enable_zb = True
        tasks.append(bwd(stage0, enable_zb=enable_zb))

    section_7 = mpmd_dim - mpmd_idx - 1
    for _ in range(section_7):
        if zero_bubble:
            tasks.append(weight_chunk())
        tasks.append(bwd(stage0, enable_zb=True))

    if zero_bubble:
        section_8 = mpmd_idx + 1
        for _ in range(section_8):
            tasks.append(weight_chunk())

    expected = {stage0: n_mubatches, stage1: n_mubatches}
    if any(bwd_counts.get(stage_id, 0) != count for stage_id, count in expected.items()) or pending_w:
        raise RuntimeError(
            "DualPipeV internal schedule error: backward/W queues did not drain "
            f"for rank={mpmd_idx}, bwd_counts={bwd_counts}, pending_w={pending_w}."
        )

    return tasks
