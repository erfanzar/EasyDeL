# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for the pipeline schedule grids."""

from __future__ import annotations

import pytest

from spectrax.runtime.schedules import (
    Action,
    DualPipeV,
    Eager1F1B,
    FusedTask,
    GPipe,
    Interleaved1F1BPlusOne,
    InterleavedGPipe,
    InterleavedH1,
    KimiK2,
    Phase,
    Std1F1B,
    ZeroBubbleH1,
)


def _count_phase(grid, n, phase):
    """Count actions of a specific phase per stage."""
    per_stage = [0] * n
    for row in grid:
        for s, cell in enumerate(row):
            if cell is not None and cell.phase == phase:
                per_stage[s] += 1
    return per_stage


def _iter_actions_in_cell(cell):
    """Iterate actions in a schedule cell."""
    if cell is None:
        return ()
    if isinstance(cell, FusedTask):
        return (cell.fwd, cell.bwd)
    return (cell,)


def _count_phase_including_fused(grid, n, phase):
    """Count actions of a specific phase per physical stage, including fused cells."""
    per_stage = [0] * n
    for row in grid:
        for s, cell in enumerate(row):
            for action in _iter_actions_in_cell(cell):
                if action.phase == phase:
                    per_stage[s] += 1
    return per_stage


def _has_cross_rank_fwd_bwd_overlap(grid):
    """Return whether any row has FWD and BWD-family work on different ranks."""
    for row in grid:
        fwd_ranks = set()
        bwd_ranks = set()
        for rank, cell in enumerate(row):
            for action in _iter_actions_in_cell(cell):
                if action.phase == Phase.FWD:
                    fwd_ranks.add(rank)
                elif action.phase in (Phase.BWD, Phase.BWD_I, Phase.BWD_W):
                    bwd_ranks.add(rank)
        if any(fwd_rank != bwd_rank for fwd_rank in fwd_ranks for bwd_rank in bwd_ranks):
            return True
    return False


def _actions_monotonic_in_microbatch(grid, n, phase):
    """Return True iff microbatch indices for ``phase`` at each stage are non-decreasing."""
    per_stage_mbs = [[] for _ in range(n)]
    for row in grid:
        for s, cell in enumerate(row):
            if cell is not None and cell.phase == phase:
                per_stage_mbs[s].append(cell.microbatch)
    return all(mbs == sorted(mbs) for mbs in per_stage_mbs)


class TestGPipe:
    """Tests for the :class:`GPipe` schedule.

    GPipe runs every microbatch's forward pass through all stages, then
    every microbatch's backward pass. Bubble fraction is ``(n-1)/(n+m-1)``.
    """

    def test_fwd_count_matches_microbatches(self):
        """Every stage processes exactly M microbatches in the forward phase."""
        g = GPipe(microbatches=4).build(n_stages=4)
        assert _count_phase(g, 4, Phase.FWD) == [4, 4, 4, 4]

    def test_bwd_count_matches_microbatches(self):
        """Every stage processes exactly M microbatches in the backward phase."""
        g = GPipe(microbatches=4).build(n_stages=4)
        assert _count_phase(g, 4, Phase.BWD) == [4, 4, 4, 4]

    def test_fwds_finish_before_any_bwd(self):
        """All forwards complete before any backward starts (possibly overlapping at the boundary)."""
        g = GPipe(microbatches=4).build(n_stages=4)
        last_fwd_t = 0
        first_bwd_t = None
        for t, row in enumerate(g):
            for cell in row:
                if cell is None:
                    continue
                if cell.phase == Phase.FWD:
                    last_fwd_t = max(last_fwd_t, t)
                elif cell.phase == Phase.BWD and first_bwd_t is None:
                    first_bwd_t = t
        assert first_bwd_t is not None
        assert first_bwd_t >= last_fwd_t

    def test_peak_activations(self):
        """Peak stashed activations equal M (one stash per microbatch)."""
        assert GPipe(microbatches=4).peak_activations(4) == 4
        assert GPipe(microbatches=16).peak_activations(4) == 16

    def test_microbatch_order_monotonic(self):
        """Microbatch order at each stage is monotonically non-decreasing in both phases."""
        g = GPipe(microbatches=4).build(n_stages=4)
        assert _actions_monotonic_in_microbatch(g, 4, Phase.FWD)
        assert _actions_monotonic_in_microbatch(g, 4, Phase.BWD)


class TestStd1F1B:
    """Tests for the standard 1F1B schedule (bounded activation memory)."""

    def test_fwd_count_matches_microbatches(self):
        """Every stage still runs M forwards."""
        g = Std1F1B(microbatches=8).build(n_stages=4)
        assert _count_phase(g, 4, Phase.FWD) == [8, 8, 8, 8]

    def test_bwd_count_matches_microbatches(self):
        """Every stage still runs M backwards."""
        g = Std1F1B(microbatches=8).build(n_stages=4)
        assert _count_phase(g, 4, Phase.BWD) == [8, 8, 8, 8]

    def test_peak_activations_bounded_by_n_stages(self):
        """Peak activations are bounded by ``n_stages`` (independent of M once M >= n)."""
        assert Std1F1B(microbatches=4).peak_activations(4) == 4
        assert Std1F1B(microbatches=64).peak_activations(4) == 4

    def test_microbatch_order_monotonic(self):
        """Microbatch order is monotonic per stage in both phases."""
        g = Std1F1B(microbatches=8).build(n_stages=4)
        assert _actions_monotonic_in_microbatch(g, 4, Phase.FWD)
        assert _actions_monotonic_in_microbatch(g, 4, Phase.BWD)

    def test_rejects_m_less_than_n(self):
        """When ``M < n_stages`` the 1F1B steady state cannot form."""
        with pytest.raises(ValueError, match="microbatches >= n_stages"):
            Std1F1B(microbatches=2).build(n_stages=4)

    def test_has_cross_rank_forward_backward_overlap(self):
        """Std1F1B must not collapse into all-forward then all-backward GPipe."""
        assert _has_cross_rank_fwd_bwd_overlap(Std1F1B(microbatches=8).build(n_stages=4))


class TestZeroBubbleH1:
    """Tests for the ZB-H1 schedule (split BWD into BWD_I + BWD_W)."""

    def test_bwd_i_count_matches_microbatches(self):
        """Every stage emits M backward-input actions."""
        g = ZeroBubbleH1(microbatches=4).build(n_stages=4)
        assert _count_phase(g, 4, Phase.BWD_I) == [4, 4, 4, 4]

    def test_bwd_w_count_matches_microbatches(self):
        """Every stage emits M backward-weight actions."""
        g = ZeroBubbleH1(microbatches=4).build(n_stages=4)
        assert _count_phase(g, 4, Phase.BWD_W) == [4, 4, 4, 4]

    def test_bwd_w_never_precedes_bwd_i(self):
        """For each (stage, microbatch), BWD_W happens strictly after its BWD_I."""
        g = ZeroBubbleH1(microbatches=4).build(n_stages=4)
        per_stage_bwd_i: dict[tuple[int, int], int] = {}
        for t, row in enumerate(g):
            for s, cell in enumerate(row):
                if cell is not None and cell.phase == Phase.BWD_I:
                    per_stage_bwd_i[(s, cell.microbatch)] = t
        for t, row in enumerate(g):
            for s, cell in enumerate(row):
                if cell is not None and cell.phase == Phase.BWD_W:
                    assert per_stage_bwd_i[(s, cell.microbatch)] < t, (
                        f"BWD_W at (t={t}, s={s}, mb={cell.microbatch}) precedes its BWD_I"
                    )

    def test_peak_activations(self):
        """Peak activations equal ``n_stages`` in ZB-H1."""
        assert ZeroBubbleH1(microbatches=4).peak_activations(4) == 4


class TestInterleavedH1:
    """Tests for the interleaved 1F1B-H1 schedule (virtual stages)."""

    def test_virtual_stages_validation(self):
        """``virtual_stages`` must be >= 1."""
        with pytest.raises(ValueError):
            InterleavedH1(microbatches=4, virtual_stages=0)

    def test_microbatch_counts_with_virtual_2(self):
        """Each physical device owns V virtual stages, so ``V*M`` forwards per device."""
        g = InterleavedH1(microbatches=4, virtual_stages=2).build(n_stages=4)
        assert _count_phase(g, 4, Phase.FWD) == [8, 8, 8, 8]

    def test_physical_scheduler_is_default(self):
        """InterleavedH1 emits the physical virtual schedule directly."""
        grid = InterleavedH1(microbatches=8, virtual_stages=2).build(n_stages=2)

        assert grid
        assert all(len(row) == 2 for row in grid)
        assert _count_phase(grid, 2, Phase.FWD) == [16, 16]

    def test_peak_activations_scales_with_virtual(self):
        """Peak activations scale linearly with ``virtual_stages``."""
        assert InterleavedH1(microbatches=4, virtual_stages=2).peak_activations(4) == 8
        assert InterleavedH1(microbatches=4, virtual_stages=4).peak_activations(4) == 16


class TestBubbleRatio:
    """Tests for ``bubble_ratio`` across schedules."""

    def test_gpipe_has_bubble(self):
        """GPipe has 2(n-1) idle slots per device at the ends, so ratio in (0, 1)."""
        r = GPipe(microbatches=4).bubble_ratio(n_stages=4)
        assert 0 < r < 1

    def test_zb_bubble_smaller_than_gpipe(self):
        """ZB-H1 fills bubbles with W-grad work, so its bubble ratio is strictly smaller than GPipe's."""
        gpipe_r = GPipe(microbatches=8).bubble_ratio(n_stages=4)
        zb_r = ZeroBubbleH1(microbatches=8).bubble_ratio(n_stages=4)
        assert zb_r < gpipe_r


class TestScheduleValidation:
    """Constructor-level validation across the core schedules."""

    def test_zero_microbatches_raises(self):
        """``microbatches=0`` is rejected by every schedule."""
        with pytest.raises(ValueError):
            GPipe(microbatches=0)
        with pytest.raises(ValueError):
            Std1F1B(microbatches=0)
        with pytest.raises(ValueError):
            ZeroBubbleH1(microbatches=0)


class TestActionMetadata:
    """Tests for the :class:`Action` dataclass metadata."""

    def test_action_is_frozen(self):
        """``Action`` is frozen: attribute assignment raises."""
        a = Action(Phase.FWD, 3)
        with pytest.raises(AttributeError):
            a.microbatch = 7

    def test_action_virtual_stage_default_zero(self):
        """``virtual_stage`` defaults to 0 when not supplied."""
        a = Action(Phase.FWD, 0)
        assert a.virtual_stage == 0


class TestEager1F1B:
    """Tests for the eager 1F1B schedule (``2n-1`` warmup)."""

    def test_fwd_count_matches_microbatches(self):
        """Every stage runs M forwards."""
        g = Eager1F1B(microbatches=8).build(n_stages=4)
        assert _count_phase(g, 4, Phase.FWD) == [8, 8, 8, 8]

    def test_bwd_count_matches_microbatches(self):
        """Every stage runs M backwards."""
        g = Eager1F1B(microbatches=8).build(n_stages=4)
        assert _count_phase(g, 4, Phase.BWD) == [8, 8, 8, 8]

    def test_microbatch_order_monotonic(self):
        """Microbatch order per stage is monotonic in both phases."""
        g = Eager1F1B(microbatches=8).build(n_stages=4)
        assert _actions_monotonic_in_microbatch(g, 4, Phase.FWD)
        assert _actions_monotonic_in_microbatch(g, 4, Phase.BWD)

    def test_peak_activations_follows_longer_warmup(self):
        """Peak activations ~ ``2n - 1`` at rank 0, capped at M."""
        assert Eager1F1B(microbatches=8).peak_activations(4) == 7
        assert Eager1F1B(microbatches=3).peak_activations(4) == 3

    def test_has_cross_rank_forward_backward_overlap(self):
        """Eager1F1B keeps true 1F1B overlap after its longer warmup."""
        assert _has_cross_rank_fwd_bwd_overlap(Eager1F1B(microbatches=8).build(n_stages=4))


class TestDualPipeV:
    """Tests for the DualPipeV V-shaped schedule."""

    def test_fwd_count_matches_two_virtuals_per_rank(self):
        """Each physical rank hosts 2 virtual stages, giving ``2*M`` forwards per rank."""
        g = DualPipeV(microbatches=8).build(n_stages=4)
        assert _count_phase_including_fused(g, 4, Phase.FWD) == [16, 16, 16, 16]

    def test_split_bwd_counts_match_two_virtuals_per_rank(self):
        """Full BWD plus split BWD_I covers ``2*M`` backward chunks per rank."""
        g = DualPipeV(microbatches=8).build(n_stages=4)
        full = _count_phase_including_fused(g, 4, Phase.BWD)
        split_i = _count_phase_including_fused(g, 4, Phase.BWD_I)
        split_w = _count_phase_including_fused(g, 4, Phase.BWD_W)
        assert [a + b for a, b in zip(full, split_i, strict=True)] == [16, 16, 16, 16]
        assert split_i == split_w

    def test_peak_activations_double_n_stages(self):
        """Peak activations equal ``2 * n_stages`` (two virtuals per rank)."""
        assert DualPipeV(microbatches=4).peak_activations(4) == 8

    def test_rejects_short_logical_1f1b_grid(self):
        """DualPipeV has ``2*n`` logical stages, so a real 1F1B grid needs ``M >= 2*n``."""
        with pytest.raises(ValueError, match=r"microbatches >= 2 \* n_stages"):
            DualPipeV(microbatches=4).build(n_stages=4)

    def test_virtual_stages_seen(self):
        """Both virtual_stage 0 and 1 appear in the emitted actions."""
        g = DualPipeV(microbatches=8).build(n_stages=4)
        virtuals_seen: set[int] = set()
        for row in g:
            for cell in row:
                for action in _iter_actions_in_cell(cell):
                    virtuals_seen.add(action.virtual_stage)
        assert virtuals_seen == {0, 1}

    def test_no_collisions_at_same_rank_time(self):
        """At each time step every rank runs at most one action."""
        g = DualPipeV(microbatches=8).build(n_stages=4)
        for row in g:
            assert len(row) == 4
            for cell in row:
                assert cell is None or isinstance(cell, (Action, FusedTask))

    def test_steady_state_pairs_fwd_with_full_bwd_across_virtuals(self):
        """DualPipeV contains the DeepSeek steady-state FWD+BWD pair across V legs."""
        g = DualPipeV(microbatches=8).build(n_stages=4)
        assert any(
            isinstance(cell, FusedTask)
            and cell.fwd.phase is Phase.FWD
            and cell.bwd.phase is Phase.BWD
            and cell.fwd.virtual_stage != cell.bwd.virtual_stage
            for row in g
            for cell in row
        )

    def test_zero_bubble_can_be_disabled_for_full_bwd_chunks(self):
        """The V pairing can run without split BWD-I/BWD-W slots on runtimes where split VJPs are costly."""
        g = DualPipeV(microbatches=8, zero_bubble=False).build(n_stages=4)
        assert _count_phase_including_fused(g, 4, Phase.FWD) == [16, 16, 16, 16]
        assert _count_phase_including_fused(g, 4, Phase.BWD) == [16, 16, 16, 16]
        assert _count_phase_including_fused(g, 4, Phase.BWD_I) == [0, 0, 0, 0]
        assert _count_phase_including_fused(g, 4, Phase.BWD_W) == [0, 0, 0, 0]


class TestInterleavedGPipe:
    """Tests for the interleaved variant of GPipe with virtual stages."""

    def test_fwd_count_matches_virtual_times_microbatches(self):
        """Each physical rank runs ``V*M`` forwards."""
        g = InterleavedGPipe(microbatches=4, virtual_stages=2).build(n_stages=4)
        assert _count_phase(g, 4, Phase.FWD) == [8, 8, 8, 8]

    def test_bwd_count_matches_virtual_times_microbatches(self):
        """Each physical rank runs ``V*M`` backwards."""
        g = InterleavedGPipe(microbatches=4, virtual_stages=2).build(n_stages=4)
        assert _count_phase(g, 4, Phase.BWD) == [8, 8, 8, 8]

    def test_virtual_stages_validation(self):
        """``virtual_stages=0`` is rejected."""
        with pytest.raises(ValueError):
            InterleavedGPipe(microbatches=4, virtual_stages=0)

    def test_peak_activations_is_virtual_times_microbatches(self):
        """Peak activations equal ``V*M`` (GPipe-like stashing per virtual)."""
        assert InterleavedGPipe(microbatches=4, virtual_stages=2).peak_activations(4) == 8

    def test_fwds_before_all_bwds(self):
        """All forwards finish before any backward starts (same GPipe ordering invariant)."""
        g = InterleavedGPipe(microbatches=4, virtual_stages=2).build(n_stages=4)
        last_fwd_t = 0
        first_bwd_t = None
        for t, row in enumerate(g):
            for cell in row:
                if cell is None:
                    continue
                if cell.phase == Phase.FWD:
                    last_fwd_t = max(last_fwd_t, t)
                elif cell.phase == Phase.BWD and first_bwd_t is None:
                    first_bwd_t = t
        assert first_bwd_t is not None
        assert first_bwd_t >= last_fwd_t

    def test_loop_and_contiguous_layouts_assign_same_logical_set(self):
        """Virtual layouts may change ownership order but never lose logical stages."""
        loop = InterleavedGPipe(microbatches=4, virtual_stages=2, stage_layout="loop")
        contiguous = InterleavedGPipe(microbatches=4, virtual_stages=2, stage_layout="contiguous")
        n = 4

        loop_order = [loop.logical_at(rank, virt, n) for rank in range(n) for virt in range(2)]
        contiguous_order = [contiguous.logical_at(rank, virt, n) for rank in range(n) for virt in range(2)]

        assert sorted(loop_order) == list(range(8))
        assert sorted(contiguous_order) == list(range(8))
        assert loop_order != contiguous_order


class TestInterleaved1F1BPlusOne:
    """Tests for the +1-warmup interleaved 1F1B variant."""

    def test_one_more_row_than_interleaved(self):
        """Grid is exactly 1 row longer than InterleavedH1 at same config."""
        base = InterleavedH1(microbatches=4, virtual_stages=2).build(n_stages=4)
        plus1 = Interleaved1F1BPlusOne(microbatches=4, virtual_stages=2).build(n_stages=4)
        assert len(plus1) == len(base) + 1

    def test_fwd_count_preserved(self):
        """Total FWD count per stage is unchanged vs. InterleavedH1."""
        plus1 = Interleaved1F1BPlusOne(microbatches=4, virtual_stages=2).build(n_stages=4)
        base = InterleavedH1(microbatches=4, virtual_stages=2).build(n_stages=4)
        assert _count_phase(plus1, 4, Phase.FWD) == _count_phase(base, 4, Phase.FWD)

    def test_bwd_count_preserved(self):
        """Total BWD count per stage is unchanged vs. InterleavedH1."""
        plus1 = Interleaved1F1BPlusOne(microbatches=4, virtual_stages=2).build(n_stages=4)
        base = InterleavedH1(microbatches=4, virtual_stages=2).build(n_stages=4)
        assert _count_phase(plus1, 4, Phase.BWD) == _count_phase(base, 4, Phase.BWD)


class TestKimiK2:
    """Tests for the Kimi-K2 warmup-bumped schedule."""

    def test_fwd_count_preserved(self):
        """KimiK2 keeps total FWD count = ``M*V`` per rank (same as InterleavedH1)."""
        kimi = KimiK2(microbatches=4, virtual_stages=2).build(n_stages=4)
        base = InterleavedH1(microbatches=4, virtual_stages=2).build(n_stages=4)
        assert _count_phase(kimi, 4, Phase.FWD) == _count_phase(base, 4, Phase.FWD)

    def test_bwd_count_preserved(self):
        """KimiK2 keeps full BWD chunks by default for the fast runtime path."""
        kimi = KimiK2(microbatches=4, virtual_stages=2).build(n_stages=4)
        base = InterleavedH1(microbatches=4, virtual_stages=2).build(n_stages=4)
        assert _count_phase(kimi, 4, Phase.BWD) == _count_phase(base, 4, Phase.BWD)
        assert _count_phase(kimi, 4, Phase.BWD_I) == [0, 0, 0, 0]
        assert _count_phase(kimi, 4, Phase.BWD_W) == [0, 0, 0, 0]

    def test_split_backward_is_explicit(self):
        """KimiK2 can opt into BWD_I/BWD_W chunks for split-backward experiments."""
        split = KimiK2(microbatches=4, virtual_stages=2, split_backward=True).build(n_stages=4)
        base = InterleavedH1(microbatches=4, virtual_stages=2).build(n_stages=4)
        assert _count_phase(split, 4, Phase.BWD_I) == _count_phase(base, 4, Phase.BWD)
        assert _count_phase(split, 4, Phase.BWD_W) == _count_phase(base, 4, Phase.BWD)
        assert _count_phase(split, 4, Phase.BWD) == [0, 0, 0, 0]

    def test_extra_warmup_is_explicit(self):
        """KimiK2 warmup is an explicit constructor field, not an env switch."""
        base = KimiK2(microbatches=8, virtual_stages=2, extra_warmup=0).build(n_stages=2)
        bumped = KimiK2(microbatches=8, virtual_stages=2, extra_warmup=1).build(n_stages=2)

        assert _count_phase(base, 2, Phase.FWD) == _count_phase(bumped, 2, Phase.FWD)
        assert _count_phase(base, 2, Phase.BWD) == _count_phase(bumped, 2, Phase.BWD)
        assert len(bumped) <= len(base)

    def test_physical_scheduler_preserves_data_dependencies(self):
        """The optimized physical schedule keeps all logical FWD/BWD dependencies."""
        n = 2
        v = 2
        m = 8
        schedule = KimiK2(microbatches=m, virtual_stages=v)
        grid = schedule.build(n_stages=n)
        n_logical = n * v
        times: dict[tuple[int, Phase, int], int] = {}

        for t, row in enumerate(grid):
            for rank, cell in enumerate(row):
                if cell is None:
                    continue
                logical = schedule.logical_at(rank, cell.virtual_stage, n)
                times[(logical, cell.phase, cell.microbatch)] = t

        for logical in range(n_logical):
            fwd_mbs = [mb for (stage, phase, mb), _t in times.items() if stage == logical and phase is Phase.FWD]
            bwd_mbs = [mb for (stage, phase, mb), _t in times.items() if stage == logical and phase is Phase.BWD]
            assert sorted(fwd_mbs) == list(range(m))
            assert sorted(bwd_mbs) == list(range(m))
            for mb in range(m):
                assert times[(logical, Phase.BWD, mb)] > times[(logical, Phase.FWD, mb)]
                if logical > 0:
                    assert times[(logical, Phase.FWD, mb)] > times[(logical - 1, Phase.FWD, mb)]
                if logical + 1 < n_logical:
                    assert times[(logical, Phase.BWD, mb)] > times[(logical + 1, Phase.BWD, mb)]


@pytest.mark.parametrize(
    ("schedule", "n_stages", "microbatches", "virtual_stages", "bwd_phases"),
    [
        (GPipe(microbatches=8), 4, 8, 1, (Phase.BWD,)),
        (Std1F1B(microbatches=8), 4, 8, 1, (Phase.BWD,)),
        (Eager1F1B(microbatches=8), 4, 8, 1, (Phase.BWD,)),
        (ZeroBubbleH1(microbatches=8), 4, 8, 1, (Phase.BWD_I, Phase.BWD_W)),
        (InterleavedH1(microbatches=8, virtual_stages=2), 4, 8, 2, (Phase.BWD,)),
        (Interleaved1F1BPlusOne(microbatches=8, virtual_stages=2), 4, 8, 2, (Phase.BWD,)),
        (InterleavedGPipe(microbatches=8, virtual_stages=2), 4, 8, 2, (Phase.BWD,)),
        (KimiK2(microbatches=8, virtual_stages=2, extra_warmup=1), 4, 8, 2, (Phase.BWD,)),
        (DualPipeV(microbatches=8), 4, 8, 2, (Phase.BWD, Phase.BWD_I)),
    ],
)
def test_all_schedulers_emit_expected_physical_work(schedule, n_stages, microbatches, virtual_stages, bwd_phases):
    """Every scheduler emits the right amount of work for each physical rank."""
    grid = schedule.build(n_stages)
    expected = [microbatches * virtual_stages] * n_stages

    assert _count_phase_including_fused(grid, n_stages, Phase.FWD) == expected
    if isinstance(schedule, DualPipeV):
        full = _count_phase_including_fused(grid, n_stages, Phase.BWD)
        split_i = _count_phase_including_fused(grid, n_stages, Phase.BWD_I)
        split_w = _count_phase_including_fused(grid, n_stages, Phase.BWD_W)
        assert [a + b for a, b in zip(full, split_i, strict=True)] == expected
        assert split_i == split_w
    else:
        for phase in bwd_phases:
            assert _count_phase_including_fused(grid, n_stages, phase) == expected


class TestFusedTask:
    """Tests for the :class:`FusedTask` paired-action container."""

    def test_split_returns_actions(self):
        """``split`` returns the stored (fwd, bwd) actions with their phases/microbatches intact."""
        f = FusedTask(Action(Phase.FWD, 3), Action(Phase.BWD, 1))
        fwd, bwd = f.split()
        assert fwd.phase == Phase.FWD and fwd.microbatch == 3
        assert bwd.phase == Phase.BWD and bwd.microbatch == 1

    def test_is_frozen(self):
        """``FusedTask`` is frozen: attribute assignment raises."""
        f = FusedTask(Action(Phase.FWD, 3), Action(Phase.BWD, 1))
        with pytest.raises(AttributeError):
            f.fwd = Action(Phase.FWD, 0)


class TestFusionHelpers:
    """Tests for :func:`fuse_1f1b_steady_state` and :func:`fuse_zerobubble_bwd_pair`."""

    def test_1f1b_fusion_reduces_dispatch_count(self):
        """Fusion on Std1F1B replaces adjacent FWD->BWD pairs with :class:`FusedTask`."""
        from spectrax.runtime.schedules import fuse_1f1b_steady_state

        sch = Std1F1B(microbatches=8)
        grid = sch.build(n_stages=4)
        total_actions = sum(1 for row in grid for c in row if c is not None)
        fused = fuse_1f1b_steady_state(grid)
        fused_pairs = sum(1 for row in fused for c in row if isinstance(c, FusedTask))
        remaining = sum(1 for row in fused for c in row if isinstance(c, Action))
        assert fused_pairs > 0
        assert total_actions == remaining + 2 * fused_pairs

    def test_1f1b_fusion_preserves_grid_shape(self):
        """Grid dimensions unchanged — only cell contents differ."""
        from spectrax.runtime.schedules import fuse_1f1b_steady_state

        grid = Std1F1B(microbatches=6).build(n_stages=3)
        fused = fuse_1f1b_steady_state(grid)
        assert len(fused) == len(grid)
        assert all(len(r1) == len(r0) for r0, r1 in zip(grid, fused, strict=False))

    def test_1f1b_fusion_does_not_cross_ranks(self):
        """Fused pairs live on the same rank — never combine actions from different ranks."""
        from spectrax.runtime.schedules import fuse_1f1b_steady_state

        grid = Std1F1B(microbatches=8).build(n_stages=4)
        fused = fuse_1f1b_steady_state(grid)
        for row in fused:
            for cell in row:
                if isinstance(cell, FusedTask):
                    assert cell.fwd.phase == Phase.FWD
                    assert cell.bwd.phase == Phase.BWD

    def test_1f1b_fusion_idempotent(self):
        """Re-fusing an already fused grid produces no further changes."""
        from spectrax.runtime.schedules import fuse_1f1b_steady_state

        grid = Std1F1B(microbatches=8).build(n_stages=4)
        f1 = fuse_1f1b_steady_state(grid)
        f2 = fuse_1f1b_steady_state(f1)
        assert f1 == f2

    def test_zb_fusion_pairs_bwd_i_bwd_w(self):
        """ZB fusion combines adjacent ``BWD_I -> BWD_W`` of the same microbatch."""
        from spectrax.runtime.schedules import fuse_zerobubble_bwd_pair

        grid = ZeroBubbleH1(microbatches=8).build(n_stages=4)
        fused = fuse_zerobubble_bwd_pair(grid)
        for row in fused:
            for cell in row:
                if isinstance(cell, FusedTask):
                    assert cell.fwd.phase == Phase.BWD_I
                    assert cell.bwd.phase == Phase.BWD_W
                    assert cell.fwd.microbatch == cell.bwd.microbatch

    def test_fusion_action_count_preserved(self):
        """Fused grid has same total action count (each FusedTask counts as 2)."""
        from spectrax.runtime.schedules import fuse_1f1b_steady_state

        grid = Std1F1B(microbatches=6).build(n_stages=3)
        orig_count = sum(1 for row in grid for c in row if c is not None)
        fused = fuse_1f1b_steady_state(grid)
        fused_count = sum(2 if isinstance(c, FusedTask) else 1 for row in fused for c in row if c is not None)
        assert orig_count == fused_count
