# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :mod:`spectrax.pipeline.per_rank` — per-rank compiled schedule programs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from spectrax.runtime.mpmd.per_rank import (
    compile_per_rank_bwd,
    compile_per_rank_fwd,
    extract_rank_actions,
    run_gpipe_per_rank,
)
from spectrax.runtime.schedules import GPipe, Phase, Std1F1B


class TestExtractRankActions:
    """Tests for :func:`extract_rank_actions`."""

    def test_gpipe_rank_0_has_fwd_then_bwd(self):
        """GPipe rank 0 does all FWDs then all BWDs in mb order."""
        sch = GPipe(microbatches=4)
        actions = extract_rank_actions(sch, rank=0, n_stages=2)
        assert len(actions) == 8
        fwd_mbs = [a.microbatch for a in actions if a.phase == Phase.FWD]
        bwd_mbs = [a.microbatch for a in actions if a.phase == Phase.BWD]
        assert fwd_mbs == [0, 1, 2, 3]
        assert bwd_mbs == [0, 1, 2, 3]

    def test_gpipe_all_ranks_same_action_count(self):
        """In GPipe each rank does 2M actions (M fwds + M bwds)."""
        sch = GPipe(microbatches=3)
        for r in range(4):
            assert len(extract_rank_actions(sch, rank=r, n_stages=4)) == 6

    def test_1f1b_interleaves_fwd_and_bwd(self):
        """Std1F1B schedule intermixes FWD and BWD on each rank."""
        sch = Std1F1B(microbatches=4)
        acts = extract_rank_actions(sch, rank=0, n_stages=2)
        phases = [a.phase for a in acts]
        assert Phase.FWD in phases
        assert Phase.BWD in phases
        assert phases[0] == Phase.FWD


class TestCompilePerRankFwd:
    """Tests for :func:`compile_per_rank_fwd`."""

    def test_fwd_program_runs_all_microbatches(self):
        """The compiled fwd program produces outputs for every mb."""
        sch = GPipe(microbatches=4)

        def fwd(params, rest, x):
            """Forward pass helper."""
            return x * params["w"]

        program = compile_per_rank_fwd(
            rank=0,
            schedule=sch,
            n_stages=2,
            microbatches=4,
            fwd_fn=fwd,
        )
        mb_inputs = jnp.arange(4 * 3, dtype=jnp.float32).reshape(4, 3)
        params = {"w": jnp.asarray(2.0)}
        mb_out, mb_saved = program(params, {}, mb_inputs)
        assert jnp.allclose(mb_out, mb_inputs * 2.0)
        assert jnp.allclose(mb_saved, mb_inputs)

    def test_fwd_program_allows_stage_output_shape_change(self):
        """Per-rank FWD stacks actual outputs instead of assuming input-shaped activations."""
        sch = GPipe(microbatches=3)

        def fwd(params, rest, x):
            """Forward pass helper."""
            del rest
            return jnp.stack([x.sum(), x.sum() * params["w"], x.mean()], axis=0)

        program = compile_per_rank_fwd(
            rank=0,
            schedule=sch,
            n_stages=2,
            microbatches=3,
            fwd_fn=fwd,
        )
        mb_inputs = jnp.arange(3 * 4, dtype=jnp.float32).reshape(3, 4)
        params = {"w": jnp.asarray(2.0)}
        mb_out, mb_saved = program(params, {}, mb_inputs)

        assert mb_out.shape == (3, 3)
        assert jnp.allclose(mb_out[:, 0], mb_inputs.sum(axis=1))
        assert jnp.allclose(mb_saved, mb_inputs)

    def test_fwd_program_rejects_non_gpipe(self):
        """Non-GPipe schedules are explicitly rejected."""
        sch = Std1F1B(microbatches=4)
        with pytest.raises(NotImplementedError, match="GPipe"):
            compile_per_rank_fwd(
                rank=0,
                schedule=sch,
                n_stages=2,
                microbatches=4,
                fwd_fn=lambda p, r, x: x,
            )


class TestCompilePerRankBwd:
    """Tests for :func:`compile_per_rank_bwd`."""

    def test_bwd_program_non_terminal(self):
        """Non-terminal rank consumes incoming cotangents and emits outgoing cots + g_params."""
        sch = GPipe(microbatches=3)

        def bwd(params, rest, x, g_y):
            """Backward pass helper."""
            return {"w": (x * g_y).sum()}, g_y * params["w"]

        program = compile_per_rank_bwd(
            rank=0,
            schedule=sch,
            n_stages=2,
            microbatches=3,
            bwd_fn=bwd,
            is_terminal=False,
        )
        saved_in = jnp.ones((3, 2), dtype=jnp.float32)
        saved_out = jnp.zeros_like(saved_in)
        cots_in = jnp.ones((3, 2), dtype=jnp.float32)
        params = {"w": jnp.asarray(3.0)}
        g_p, cots_out, loss = program(params, {}, saved_in, saved_out, cots_in)
        assert jnp.allclose(cots_out, 3.0)
        assert float(loss) == 0.0
        assert jnp.allclose(g_p["w"], 6.0)

    def test_bwd_program_terminal_requires_loss_fn(self):
        """Terminal rank without a ``loss_and_g_y`` raises."""
        sch = GPipe(microbatches=2)
        with pytest.raises(ValueError, match="loss_and_g_y"):
            compile_per_rank_bwd(
                rank=1,
                schedule=sch,
                n_stages=2,
                microbatches=2,
                bwd_fn=lambda p, r, x, g: ({}, g),
                is_terminal=True,
            )


class TestRunGpipePerRank:
    """End-to-end correctness test for :func:`run_gpipe_per_rank`.

    Uses a toy 2-stage linear model where each stage is a scalar
    multiply. Compares the output loss and parameter gradients to a
    hand-rolled reference.
    """

    def test_matches_hand_computed_reference(self):
        """Toy 2-stage scalar pipeline matches the analytical result."""
        sch = GPipe(microbatches=4)

        def fwd_fn(params, rest, x):
            """Forward function helper."""
            return x * params["w"]

        def bwd_fn(params, rest, x, g_y):
            """Backward function helper."""
            g_params = {"w": (x * g_y).sum()}
            g_x = g_y * params["w"]
            return g_params, g_x

        def loss_and_g_y(y, tgt):
            """Compute the loss."""
            diff = y - tgt
            loss = 0.5 * jnp.sum(diff * diff)
            return loss, diff

        M = 4
        xs = jnp.arange(M * 2, dtype=jnp.float32).reshape(M, 2)
        tgt = jnp.ones((M, 2), dtype=jnp.float32)
        stage_params = [{"w": jnp.asarray(2.0)}, {"w": jnp.asarray(3.0)}]

        dev = jax.devices()[0]
        shardings = [dev, dev]

        mean_loss, grads = run_gpipe_per_rank(
            n_stages=2,
            microbatches=M,
            schedule=sch,
            fwd_fns=[fwd_fn, fwd_fn],
            bwd_fns=[bwd_fn, bwd_fn],
            loss_and_g_y=loss_and_g_y,
            stage_params=stage_params,
            stage_rest=[{}, {}],
            stage_shardings=shardings,
            xs=xs,
            target_args=(tgt,),
        )

        w0, w1 = 2.0, 3.0
        y0 = xs * w0
        y1 = y0 * w1
        diff = y1 - tgt
        ref_loss = 0.5 * jnp.sum(diff * diff) / M
        assert jnp.allclose(mean_loss, ref_loss, atol=1e-4)

        g_w1_ref = (y0 * diff).sum() / M
        g_y0_for_stage0 = diff * w1
        g_w0_ref = (xs * g_y0_for_stage0).sum() / M
        assert jnp.allclose(grads[0]["w"], g_w0_ref, atol=1e-4)
        assert jnp.allclose(grads[1]["w"], g_w1_ref, atol=1e-4)


class TestRejectsUnsupportedSchedules:
    """Driver refuses non-GPipe schedules with a clear error."""

    def test_std1f1b_rejected_by_compile_fwd(self):
        """Std1F1B fails at compile_per_rank_fwd."""
        with pytest.raises(NotImplementedError):
            compile_per_rank_fwd(
                rank=0,
                schedule=Std1F1B(microbatches=4),
                n_stages=2,
                microbatches=4,
                fwd_fn=lambda p, r, x: x,
            )
