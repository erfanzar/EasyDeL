# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end tests for :func:`make_scheduled_body` — shard_map scheduled pipeline."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from spectrax.runtime.schedules import GPipe, Std1F1B, ZeroBubbleH1
from spectrax.runtime.spmd.shard_map import make_scheduled_body


def _toy_stage_fn(params, x):
    """Simple scalar-weight stage: ``y = x * w``."""
    return x * params["w"]


def _toy_bwd_fn(params, x, g_y):
    """VJP of ``_toy_stage_fn``: ``g_w = sum(x * g_y)``, ``g_x = g_y * w``."""
    g_w = (x * g_y).sum()
    g_x = g_y * params["w"]
    return {"w": g_w}, g_x


def _mse_loss_and_g_y(y, tgt):
    """Return ``(0.5 * sum((y - tgt)**2), y - tgt)`` — MSE and its cotangent."""
    diff = y - tgt
    loss = 0.5 * jnp.sum(diff * diff)
    return loss, diff


def _run_shardmap_scheduled(schedule, m, per_rank_params, xs, tgt, *, use_scan=False):
    """Build and run a 2-rank scheduled shard_map step against the toy model.

    Returns ``(loss, grads_per_rank)`` — stacked parameters turn into a
    per-rank (pp, *) pytree internally; we reshape on the way in and out.
    """
    n = 2
    devs = jax.devices()[:n]
    mesh = Mesh(np.array(devs), ("pp",))
    pp_axis = "pp"

    body = make_scheduled_body(
        schedule=schedule,
        n_stages=n,
        microbatches=m,
        pp_axis=pp_axis,
        fwd_fn=_toy_stage_fn,
        bwd_fn=_toy_bwd_fn,
        loss_and_g_y=_mse_loss_and_g_y,
        mode="train",
        use_scan=use_scan,
    )

    stacked_params = {"w": jnp.stack([per_rank_params[r]["w"] for r in range(n)], axis=0)}

    params_spec = {"w": PartitionSpec(pp_axis)}
    xs_spec = PartitionSpec()
    tgt_spec = PartitionSpec()
    out_specs = (PartitionSpec(), {"w": PartitionSpec(pp_axis)})

    smap = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(params_spec, xs_spec, tgt_spec),
        out_specs=out_specs,
        axis_names=frozenset({pp_axis}),
        check_vma=False,
    )
    step = jax.jit(smap)
    with mesh:
        stacked_params_p = jax.device_put(
            stacked_params,
            {"w": NamedSharding(mesh, params_spec["w"])},
        )
        xs_p = jax.device_put(xs, NamedSharding(mesh, xs_spec))
        tgt_p = jax.device_put(tgt, NamedSharding(mesh, tgt_spec))
        loss, grads = step(stacked_params_p, xs_p, tgt_p)
    return loss, grads


def _analytic_reference(w0, w1, xs, tgt, m):
    """Hand-computed loss + ``(g_w0, g_w1)`` for a 2-stage ``y = x*w0*w1`` pipeline."""
    y0 = xs * w0
    y1 = y0 * w1
    diff = y1 - tgt
    total_loss = 0.5 * jnp.sum(diff * diff)
    mean_loss = total_loss / m
    g_w1 = (y0 * diff).sum()
    g_w0 = (xs * diff * w1).sum()
    return mean_loss, g_w0, g_w1


class TestScheduledBodyGPipe:
    """Correctness for :func:`make_scheduled_body` with :class:`GPipe`."""

    def test_gpipe_matches_analytic_reference(self):
        """Toy 2-rank pipeline under GPipe produces the analytical loss and grads."""
        m = 4
        sch = GPipe(microbatches=m)
        per_mb = 2
        xs = jnp.arange(m * per_mb, dtype=jnp.float32).reshape(m, per_mb) + 1.0
        tgt = jnp.ones((m, per_mb), dtype=jnp.float32)
        per_rank_params = [{"w": jnp.asarray(2.0)}, {"w": jnp.asarray(3.0)}]

        loss, grads = _run_shardmap_scheduled(sch, m, per_rank_params, xs, tgt)

        ref_loss, ref_g_w0, ref_g_w1 = _analytic_reference(2.0, 3.0, xs, tgt, m)
        assert jnp.allclose(loss, ref_loss, atol=1e-3)
        assert grads["w"].shape == (2,)
        assert jnp.allclose(grads["w"][0], ref_g_w0, atol=1e-3)
        assert jnp.allclose(grads["w"][1], ref_g_w1, atol=1e-3)


class TestScheduledBodyStd1F1B:
    """Correctness for :func:`make_scheduled_body` with :class:`Std1F1B`.

    Std1F1B's steady state has rank 0 doing FWD while rank 1 does BWD
    on different microbatches — the place where cross-rank compute
    overlap actually materializes. Equal result to GPipe's analytic
    reference confirms ordering semantics (forward deps, backward deps,
    ppermute direction) are correct under an interleaved schedule.
    """

    def test_std1f1b_matches_analytic_reference(self):
        """Std1F1B loss + grads match the analytical two-stage reference."""
        m = 4
        sch = Std1F1B(microbatches=m)
        per_mb = 2
        xs = jnp.arange(m * per_mb, dtype=jnp.float32).reshape(m, per_mb) + 1.0
        tgt = jnp.ones((m, per_mb), dtype=jnp.float32)
        per_rank_params = [{"w": jnp.asarray(2.0)}, {"w": jnp.asarray(3.0)}]

        loss, grads = _run_shardmap_scheduled(sch, m, per_rank_params, xs, tgt)

        ref_loss, ref_g_w0, ref_g_w1 = _analytic_reference(2.0, 3.0, xs, tgt, m)
        assert jnp.allclose(loss, ref_loss, atol=1e-3)
        assert jnp.allclose(grads["w"][0], ref_g_w0, atol=1e-3)
        assert jnp.allclose(grads["w"][1], ref_g_w1, atol=1e-3)


class TestScheduledBodyZeroBubble:
    """Correctness for split BWD_I/BWD_W schedules."""

    def test_zero_bubble_h1_matches_analytic_reference(self):
        """BWD_I sends input grads and BWD_W accumulates weight grads exactly once."""
        m = 4
        sch = ZeroBubbleH1(microbatches=m)
        per_mb = 2
        xs = jnp.arange(m * per_mb, dtype=jnp.float32).reshape(m, per_mb) + 1.0
        tgt = jnp.ones((m, per_mb), dtype=jnp.float32)
        per_rank_params = [{"w": jnp.asarray(2.0)}, {"w": jnp.asarray(3.0)}]

        loss, grads = _run_shardmap_scheduled(sch, m, per_rank_params, xs, tgt)

        ref_loss, ref_g_w0, ref_g_w1 = _analytic_reference(2.0, 3.0, xs, tgt, m)
        assert jnp.allclose(loss, ref_loss, atol=1e-3)
        assert jnp.allclose(grads["w"][0], ref_g_w0, atol=1e-3)
        assert jnp.allclose(grads["w"][1], ref_g_w1, atol=1e-3)


class TestScheduledBodyRejections:
    """Builder rejects unsupported inputs with a clear error."""

    def test_forward_mode_not_supported(self):
        """``mode='forward'`` falls outside this builder's scope (use GPipe primitive instead)."""
        with pytest.raises(NotImplementedError, match="mode='train'"):
            make_scheduled_body(
                schedule=GPipe(microbatches=2),
                n_stages=2,
                microbatches=2,
                pp_axis="pp",
                fwd_fn=_toy_stage_fn,
                bwd_fn=_toy_bwd_fn,
                loss_and_g_y=_mse_loss_and_g_y,
                mode="forward",
            )


def _analytic_reference_vstage(weights, xs, tgt, m):
    """Hand-computed loss + grads for an N=2 V=2 4-logical-stage scalar pipeline.

    ``weights`` is ``((w00, w01), (w10, w11))`` — ``weights[rank][virt]``.
    Under :class:`InterleavedH1` the logical order is
    ``w00 -> w10 -> w01 -> w11``.
    """
    (w00, w01), (w10, w11) = weights
    prod = w00 * w10 * w01 * w11
    y_final = xs * prod
    diff = y_final - tgt
    total_loss = 0.5 * jnp.sum(diff * diff)
    mean_loss = total_loss / m
    y0 = xs * w00
    y1 = y0 * w10
    y2 = y1 * w01
    g_w11 = (y2 * diff).sum()
    g_y2 = diff * w11
    g_w01 = (y1 * g_y2).sum()
    g_y1 = g_y2 * w01
    g_w10 = (y0 * g_y1).sum()
    g_y0 = g_y1 * w10
    g_w00 = (xs * g_y0).sum()
    return mean_loss, {(0, 0): g_w00, (0, 1): g_w01, (1, 0): g_w10, (1, 1): g_w11}


def _run_shardmap_virtual(schedule, m, per_rank_virt_params, xs, tgt, *, use_scan=False):
    """Run a 2-rank, V=2 shard_map pipeline with scalar-weight stages.

    ``per_rank_virt_params[r][v]`` is a dict ``{"w": jnp.asarray(...)}``
    — the stage's parameters at physical rank ``r``, virt ``v``. They are
    stacked to shape ``(pp=2, V=2)`` before entering the body.
    """
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    n = 2
    V = 2
    devs = jax.devices()[:n]
    mesh = Mesh(np.array(devs), ("pp",))
    pp_axis = "pp"

    body = make_scheduled_body(
        schedule=schedule,
        n_stages=n,
        microbatches=m,
        pp_axis=pp_axis,
        fwd_fn=_toy_stage_fn,
        bwd_fn=_toy_bwd_fn,
        loss_and_g_y=_mse_loss_and_g_y,
        mode="train",
        use_scan=use_scan,
    )

    stacked_w = jnp.stack([jnp.stack([per_rank_virt_params[r][v]["w"] for v in range(V)]) for r in range(n)])
    stacked_params = {"w": stacked_w}

    params_spec = {"w": PartitionSpec(pp_axis)}
    xs_spec = PartitionSpec()
    tgt_spec = PartitionSpec()
    out_specs = (PartitionSpec(), {"w": PartitionSpec(pp_axis)})

    smap = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(params_spec, xs_spec, tgt_spec),
        out_specs=out_specs,
        axis_names=frozenset({pp_axis}),
        check_vma=False,
    )
    step = jax.jit(smap)
    with mesh:
        stacked_params_p = jax.device_put(
            stacked_params,
            {"w": NamedSharding(mesh, params_spec["w"])},
        )
        xs_p = jax.device_put(xs, NamedSharding(mesh, xs_spec))
        tgt_p = jax.device_put(tgt, NamedSharding(mesh, tgt_spec))
        loss, grads = step(stacked_params_p, xs_p, tgt_p)
    return loss, grads


class TestScheduledBodyVirtualStages:
    """Correctness for :func:`make_scheduled_body` on virtual-stage schedules.

    :class:`InterleavedH1` at ``N=2 V=2`` hosts 4 logical stages on 2
    physical ranks: rank ``r`` holds virt ``0`` and virt ``1``. The test
    walks a simple 4-stage scalar-weight pipeline and checks the loss
    plus all 4 per-virt gradients against the hand-computed reference.
    """

    def test_interleaved_h1_matches_analytic_reference(self):
        """InterleavedH1 V=2 loss + per-virt grads match the 4-logical-stage reference."""
        from spectrax.runtime.schedules import InterleavedH1

        m = 8
        sch = InterleavedH1(microbatches=m, virtual_stages=2)
        per_mb = 2
        xs = jnp.arange(m * per_mb, dtype=jnp.float32).reshape(m, per_mb) + 1.0
        tgt = jnp.ones((m, per_mb), dtype=jnp.float32)

        weights = ((2.0, 5.0), (3.0, 7.0))
        per_rank_virt_params = [[{"w": jnp.asarray(weights[r][v])} for v in range(2)] for r in range(2)]

        loss, grads = _run_shardmap_virtual(sch, m, per_rank_virt_params, xs, tgt)

        ref_loss, ref_grads = _analytic_reference_vstage(weights, xs, tgt, m)
        assert jnp.allclose(loss, ref_loss, atol=1e-3, rtol=1e-4)
        assert grads["w"].shape == (2, 2)
        assert jnp.allclose(grads["w"][0, 0], ref_grads[(0, 0)], atol=1e-3, rtol=1e-4)
        assert jnp.allclose(grads["w"][0, 1], ref_grads[(0, 1)], atol=1e-3, rtol=1e-4)
        assert jnp.allclose(grads["w"][1, 0], ref_grads[(1, 0)], atol=1e-3, rtol=1e-4)
        assert jnp.allclose(grads["w"][1, 1], ref_grads[(1, 1)], atol=1e-3, rtol=1e-4)

    def test_interleaved_h1_scan_path_uses_schedule_logical_layout(self):
        """The scan lowering must respect ``schedule.logical_at`` for loop-layout virtual stages."""
        from spectrax.runtime.schedules import InterleavedH1

        m = 8
        sch = InterleavedH1(microbatches=m, virtual_stages=2, stage_layout="loop")
        per_mb = 2
        xs = jnp.arange(m * per_mb, dtype=jnp.float32).reshape(m, per_mb) + 1.0
        tgt = jnp.ones((m, per_mb), dtype=jnp.float32)

        weights = ((2.0, 5.0), (3.0, 7.0))
        per_rank_virt_params = [[{"w": jnp.asarray(weights[r][v])} for v in range(2)] for r in range(2)]

        loss, grads = _run_shardmap_virtual(sch, m, per_rank_virt_params, xs, tgt, use_scan=True)

        ref_loss, ref_grads = _analytic_reference_vstage(weights, xs, tgt, m)
        assert jnp.allclose(loss, ref_loss, atol=1e-3, rtol=1e-4)
        for (r, v), ref in ref_grads.items():
            assert jnp.allclose(grads["w"][r, v], ref, atol=1e-3, rtol=1e-4)

    def test_kimik2_matches_analytic_reference(self):
        """KimiK2 (InterleavedH1 + warmup+1) produces the same analytic result."""
        from spectrax.runtime.schedules import KimiK2

        m = 8
        sch = KimiK2(microbatches=m, virtual_stages=2)
        per_mb = 2
        xs = jnp.arange(m * per_mb, dtype=jnp.float32).reshape(m, per_mb) + 1.0
        tgt = jnp.ones((m, per_mb), dtype=jnp.float32)

        weights = ((2.0, 5.0), (3.0, 7.0))
        per_rank_virt_params = [[{"w": jnp.asarray(weights[r][v])} for v in range(2)] for r in range(2)]

        loss, grads = _run_shardmap_virtual(sch, m, per_rank_virt_params, xs, tgt)

        ref_loss, ref_grads = _analytic_reference_vstage(weights, xs, tgt, m)
        assert jnp.allclose(loss, ref_loss, atol=1e-3, rtol=1e-4)
        for (r, v), ref in ref_grads.items():
            assert jnp.allclose(grads["w"][r, v], ref, atol=1e-3, rtol=1e-4), (
                f"mismatch at ({r},{v}): got {grads['w'][r, v]}, ref {ref}"
            )


class TestScheduledBodyVirtualStageSmoke:
    """Smoke tests — every remaining virtual-stage schedule compiles and runs finite.

    We don't re-derive an analytical reference for every schedule
    variant; instead we verify: (a) the body compiles, (b) the
    per-rank param gradients are finite (not NaN/Inf), (c) the loss
    is positive and finite. Any silent correctness drift relative to
    the InterleavedH1/KimiK2 reference would show as NaN cotangents
    from a mismatched ppermute perm or a saved-input slot that was
    never populated.
    """

    @pytest.mark.parametrize(
        "sched_name,extra_kwargs",
        [
            ("DualPipeV", {}),
            ("Interleaved1F1BPlusOne", {"virtual_stages": 2}),
            ("InterleavedGPipe", {"virtual_stages": 2}),
        ],
    )
    def test_virtual_stage_variant_runs_finite(self, sched_name, extra_kwargs):
        """The named virtual-stage schedule runs to completion with finite outputs."""
        import spectrax.runtime.schedules as sch_mod

        cls = getattr(sch_mod, sched_name)
        m = 8
        sch = cls(microbatches=m, **extra_kwargs)
        per_mb = 2
        xs = jnp.arange(m * per_mb, dtype=jnp.float32).reshape(m, per_mb) + 1.0
        tgt = jnp.ones((m, per_mb), dtype=jnp.float32)
        weights = ((1.0, 1.0), (1.0, 1.0))
        per_rank_virt_params = [[{"w": jnp.asarray(weights[r][v])} for v in range(2)] for r in range(2)]

        try:
            loss, grads = _run_shardmap_virtual(sch, m, per_rank_virt_params, xs, tgt)
        except NotImplementedError as exc:
            pytest.skip(f"{sched_name} unsupported under make_scheduled_body: {exc}")

        assert jnp.isfinite(loss), f"{sched_name} loss is non-finite: {float(loss)}"
        assert float(loss) > 0
        assert jnp.all(jnp.isfinite(grads["w"])), f"{sched_name} per-virt grads contain NaN/Inf: {grads['w']}"
