# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Core schedule primitives shared by every concrete schedule.

Defines the four building blocks every spectrax pipeline schedule
emits or inherits:

* :class:`Phase` — enum of action kinds (``FWD``, ``BWD``, ``BWD_I``,
  ``BWD_W``).
* :class:`Action` — one ``(phase, microbatch, virtual_stage)`` event
  scheduled on one stage at one time step.
* :class:`FusedTask` — a steady-state pair of actions a runtime can
  dispatch as a single jitted call (1F1B's ``FWD/BWD`` pair, or
  zero-bubble's ``BWD_I/BWD_W`` pair).
* :class:`Schedule` — the ABC that concrete schedules subclass; it
  holds ``microbatches`` and the optional ``lazy_bwd_batching`` flag,
  plus default implementations of the virtual-stage hooks
  (:meth:`Schedule.virtual_stages_per_rank`,
  :meth:`Schedule.logical_at`, :meth:`Schedule.next_logical_loc`,
  :meth:`Schedule.terminal_loc`) for flat (one-logical-per-rank)
  schedules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class Phase(Enum):
    """Action phase: what the stage is computing.

    * :attr:`FWD` — forward pass on a microbatch's activations.
    * :attr:`BWD` — full backward pass (both input-grad and
      weight-grad fused).
    * :attr:`BWD_I` — input-grad only (zero-bubble split).
    * :attr:`BWD_W` — weight-grad only (zero-bubble split).
    """

    FWD = "fwd"
    BWD = "bwd"
    BWD_I = "bwd_i"
    BWD_W = "bwd_w"


_DEFAULT_LATENCIES: dict[Phase, int] = {
    Phase.FWD: 2,
    Phase.BWD: 4,
    Phase.BWD_I: 2,
    Phase.BWD_W: 2,
}


@dataclass(frozen=True)
class Action:
    """Single scheduled action for one stage at one time step.

    Attributes:
        phase: What the stage does (see :class:`Phase`).
        microbatch: Which microbatch the action operates on.
        virtual_stage: For :class:`InterleavedH1`, which of the
            device's multiple virtual stages runs. Zero for other
            schedules.
        latency: Relative cost weight for latency-aware scheduling.
            Defaults to the phase's default latency (FWD=2, BWD=4,
            BWD_I=BWD_W=2). Used by schedule optimizers that want to
            minimize critical-path length; the absolute value is
            arbitrary (only the ratio between phases matters).
    """

    phase: Phase
    microbatch: int
    virtual_stage: int = 0
    latency: int | None = None

    def __post_init__(self) -> None:
        """Populate :attr:`latency` from the phase's default if not provided.

        Frozen dataclasses can't assign to fields normally, so the
        update is performed via :func:`object.__setattr__`. Callers
        that pass an explicit ``latency`` keep their override; only
        ``None`` triggers the lookup.
        """
        if self.latency is None:
            object.__setattr__(self, "latency", _DEFAULT_LATENCIES.get(self.phase, 1))


@dataclass(frozen=True)
class FusedTask:
    """A steady-state action that combines a forward and a backward.

    Used by 1F1B-family schedules to let a runtime fuse the
    forward of microbatch ``a`` and the backward of microbatch ``b``
    into a single compiled XLA kernel, reducing dispatch overhead
    and improving register reuse. Runtimes that don't support fusion
    can treat a :class:`FusedTask` as two sequential :class:`Action`
    objects via :meth:`split`.

    Attributes:
        fwd: The forward half.
        bwd: The backward half.
        virtual_stage: Shared virtual stage index.
    """

    fwd: Action
    bwd: Action
    virtual_stage: int = 0

    def split(self) -> tuple[Action, Action]:
        """Return the underlying ``(fwd, bwd)`` :class:`Action` pair.

        Runtimes that don't implement kernel fusion treat a
        :class:`FusedTask` as two sequential actions and call
        :meth:`split` to recover the original pair.

        Returns:
            ``(self.fwd, self.bwd)`` — the two halves of the fused
            task in the order they would have run unfused.
        """
        return (self.fwd, self.bwd)


@dataclass
class Schedule(ABC):
    """Abstract base for pipeline schedules.

    A schedule maps time step x physical stage to a :class:`Action`
    (or ``None`` for idle). The :meth:`build` method returns the full
    2D plan; :meth:`peak_activations` reports the worst-case number of
    activation tensors any single stage holds at once (a proxy for
    peak memory).

    Virtual-stage schedules (Interleaved*, KimiK2, DualPipeV) assign
    multiple *logical* stages to each physical rank. The runtime needs
    three schedule-specific pieces of information to execute them:

    * :meth:`virtual_stages_per_rank` — how many logical stages each
      rank hosts. Flat schedules return 1.
    * :meth:`logical_at` — given ``(rank, virt)``, which logical
      stage's parameters live there.
    * :meth:`next_logical_loc` — after this ``(rank, virt)`` produces
      an activation, which ``(rank, virt)`` consumes it next (or
      ``None`` if this is the terminal stage).

    Defaults on the base class assume a flat (one-logical-per-rank)
    pipeline; virtual schedules override.

    Attributes:
        microbatches: Number of microbatches per global batch.
        lazy_bwd_batching: When ``True``, the MPMD runtime collects all
            backward actions for each logical stage during the forward
            grid walk, then dispatches them in a single vmapped backward
            per stage at the end. This reduces dispatch count (fewer
            Python->XLA round-trips) at the cost of higher peak
            activation memory because all saved inputs/outputs must be
            retained until the final batched backward. Default ``False``.
    """

    microbatches: int
    lazy_bwd_batching: bool = False

    def __post_init__(self) -> None:
        """Validate that :attr:`microbatches` is at least 1.

        Concrete subclasses are free to override and add more checks
        (e.g. :class:`InterleavedH1` validates ``virtual_stages``);
        they should call ``super().__post_init__()`` first.

        Raises:
            ValueError: If :attr:`microbatches` is less than 1.
        """
        if self.microbatches < 1:
            raise ValueError(f"Schedule.microbatches must be >= 1, got {self.microbatches}.")

    @abstractmethod
    def build(self, n_stages: int) -> list[list[Action | None]]:
        """Return the ``(T, n_stages)`` action grid for ``n_stages`` stages.

        ``result[t][s]`` is the action performed by stage ``s`` at
        time step ``t``, or ``None`` if the stage is idle at ``t``.

        Args:
            n_stages: Number of physical pipeline stages (matches the
                mesh's pipeline-axis size).

        Returns:
            A list of ``T`` rows, each a list of ``n_stages``
            :class:`Action` (or ``None``) entries.
        """

    def virtual_stages_per_rank(self) -> int:
        """Number of logical stages each physical rank hosts.

        Flat schedules return ``1``; virtual-stage schedules
        (:class:`InterleavedH1`, :class:`KimiK2`, :class:`DualPipeV`,
                ...) override.

        Returns:
            Result described by this helper.
        """
        return 1

    def logical_at(self, rank: int, virt: int, n_stages: int) -> int:
        """Return the logical-stage index hosted at ``(rank, virt)``.

        For flat schedules ``virt`` is always ``0`` and the logical
        stage equals the physical rank. Virtual-stage schedules
        override to express layouts like contiguous chunks
        (``rank * V + virt``) or strided interleaving
        (``virt * n_stages + rank``).

        Args:
            rank: Physical rank index in ``[0, n_stages)``.
            virt: Virtual-stage index in
                ``[0, virtual_stages_per_rank())``.
            n_stages: Number of physical pipeline ranks.

        Returns:
            The logical-stage index in ``[0, n_logical)``.
        """
        return rank

    def next_logical_loc(self, rank: int, virt: int, n_stages: int) -> tuple[int, int] | None:
        """Return the ``(rank, virt)`` of the downstream logical stage.

        Used by the runtime to know where to ``ppermute`` an
        activation after this stage produces it. Flat schedules send
        to ``(rank + 1, 0)`` and stop at ``rank == n_stages - 1``.

        Args:
            rank: This action's physical rank.
            virt: This action's virtual-stage index.
            n_stages: Number of physical pipeline ranks.

        Returns:
            The downstream ``(rank, virt)`` location, or ``None`` if
            this position is terminal — its output feeds ``loss_fn``
            directly.
        """
        if rank + 1 < n_stages:
            return (rank + 1, 0)
        return None

    def terminal_loc(self, n_stages: int) -> tuple[int, int]:
        """Return the ``(rank, virt)`` hosting the final logical stage.

        The runtime uses this to know which stage runs ``loss_fn``
        and seeds the backward cotangent. For flat schedules this is
        ``(n_stages - 1, 0)``; virtual schedules override.

        Args:
            n_stages: Number of physical pipeline ranks.

        Returns:
            ``(rank, virt)`` location of the terminal logical stage.
        """
        return (n_stages - 1, 0)

    @abstractmethod
    def peak_activations(self, n_stages: int) -> int:
        """Return the worst-case number of live activations per stage.

        A diagnostic used to reason about peak memory. The worst-case
        stage (typically stage 0 for GPipe, the middle stage for
        1F1B) holds this many saved activations at its memory peak.

        Args:
            n_stages: Number of physical pipeline stages.

        Returns:
            Integer upper bound on simultaneous live activation
            tensors per stage.
        """

    def total_steps(self, n_stages: int) -> int:
        """Total number of time steps the schedule occupies.

        Calls :meth:`build` and returns the number of rows. Useful
        for sizing buffers and for runtimes that want to compare
        schedules without inspecting the full grid.

        Args:
            n_stages: Number of physical pipeline ranks.

        Returns:
            The number of time-step rows in the schedule.
        """
        return len(self.build(n_stages))

    def bubble_ratio(self, n_stages: int) -> float:
        """Fraction of ``(stage x time)`` slots that are idle.

        ``0.0`` means perfectly packed; ``1.0`` means nothing runs.
        Useful for comparing schedules at the same
        ``(n_stages, microbatches)`` — e.g. seeing that
        :class:`ZeroBubbleH1` shrinks the bubble of :class:`Std1F1B`,
        or that :class:`InterleavedH1` shrinks both with virtual
        stages.

        Args:
            n_stages: Number of physical pipeline ranks.

        Returns:
            Idle fraction in ``[0.0, 1.0]``.
        """
        grid = self.build(n_stages)
        total = len(grid) * n_stages
        idle = sum(1 for row in grid for cell in row if cell is None)
        return idle / total
