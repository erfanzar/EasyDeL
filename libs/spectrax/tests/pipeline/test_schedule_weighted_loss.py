# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scheduled MPMD per-microbatch loss weighting and aux-metric parity tests.

The scheduled runtime historically reduced per-microbatch losses (and scaled
their gradients) with a uniform ``1/M`` — a mean-of-means. When each
microbatch's loss is itself a weighted mean with a *different* denominator
(SFT valid-token counts), that estimator diverges from the single-program
global weighted mean. ``sxvalue_and_grad(..., microbatch_weight_fn=...)``
must reproduce the global estimator exactly; the uniform behavior must stay
bit-compatible for callers that do not opt in.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh
from spectrax.runtime.mpmd import sxjit, sxstage_iter, sxvalue_and_grad
from spectrax.runtime.schedules import GPipe, Std1F1B
from spectrax.runtime.types import MpMdMesh

_N = 2
_D = 4
_M = 2
_BATCH = 4


@pytest.fixture(scope="module")
def mesh():
    """Two-rank MPMD mesh."""
    devs = jax.devices()[:_N]
    if len(devs) < _N:
        pytest.skip(f"need {_N} devices; have {len(devs)}")
    return MpMdMesh(Mesh(devs, axis_names=("pp",)), "pp")


@pytest.fixture(scope="module")
def wdata():
    """Params + batch with per-row masks whose valid counts differ across microbatches."""
    key = jax.random.PRNGKey(3)
    k0, k1, k2, k3 = jax.random.split(key, 4)
    w0 = jax.random.normal(k0, (_D, _D)) * 0.5
    b0 = jnp.zeros((_D,))
    w1 = jax.random.normal(k1, (_D, _D)) * 0.5
    b1 = jnp.zeros((_D,))
    x = jax.random.normal(k2, (_BATCH, _D))
    y = jax.random.normal(k3, (_BATCH, _D))
    # Microbatch 0 (rows 0-1) has 7 valid entries, microbatch 1 (rows 2-3) has 2.
    mask = jnp.asarray(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    return w0, b0, w1, b1, x, y, mask


def _forward(w0, b0, w1, b1, x):
    """Two-layer reference forward."""
    h = jnp.maximum(x @ w0 + b0, 0)
    return h @ w1 + b1


def _global_weighted_loss(w0, b0, w1, b1, x, y, mask):
    """Single-program token-weighted mean over the WHOLE batch (the SPMD estimator)."""
    out = _forward(w0, b0, w1, b1, x)
    return jnp.sum(((out - y) ** 2) * mask) / jnp.sum(mask)


def _mb_weighted_loss(w0, b0, w1, b1, x, y, mask):
    """Per-microbatch token-weighted mean (what the scheduled loss closure computes)."""
    out = _forward(w0, b0, w1, b1, x)
    return jnp.sum(((out - y) ** 2) * mask) / jnp.sum(mask)


def _weight_fn(w0, b0, w1, b1, x, y, mask):
    """Per-microbatch loss weight: the valid-token count."""
    del w0, b0, w1, b1, x, y
    return jnp.sum(mask)


def _make_scheduled(mesh, schedule, *, with_aux: bool = False):
    """Build the two-stage scheduled loss function."""

    @sxjit(mesh=mesh, schedule=schedule, batch_argnums=(4, 5, 6))
    def pipe_loss(w0, b0, w1, b1, x, y, mask):
        """Two-stage pipelined weighted-mean loss."""
        h = jnp.maximum(x @ w0 + b0, 0)
        h = sxstage_iter(h)
        out = h @ w1 + b1
        loss = jnp.sum(((out - y) ** 2) * mask) / jnp.sum(mask)
        if with_aux:
            sq_err = jnp.sum(((out - y) ** 2) * mask) / jnp.sum(mask)
            return loss, {"sq_err": sq_err, "count": jnp.sum(mask)}
        return loss

    return pipe_loss


@pytest.mark.parametrize("schedule_cls", [GPipe, Std1F1B])
def test_weighted_scheduled_loss_and_grads_match_global_estimator(mesh, wdata, schedule_cls):
    """microbatch_weight_fn reproduces the single-program weighted mean + grads."""
    ref_loss, ref_grads = jax.value_and_grad(_global_weighted_loss, argnums=(0, 1, 2, 3))(*wdata)

    pipe_loss = _make_scheduled(mesh, schedule_cls(microbatches=_M))
    loss, grads = sxvalue_and_grad(pipe_loss, argnums=(0, 1, 2, 3), microbatch_weight_fn=_weight_fn)(*wdata)

    assert jnp.allclose(loss, ref_loss, atol=1e-5, rtol=1e-5), (loss, ref_loss)
    for got, ref in zip(grads, ref_grads, strict=True):
        assert jnp.allclose(got, ref, atol=1e-5, rtol=1e-5)


def test_unweighted_scheduled_loss_keeps_uniform_mean(mesh, wdata):
    """Without a weight fn the historical uniform microbatch mean is preserved.

    Also proves the estimators actually differ on this data — the
    falsifying condition the weighted path fixes.
    """
    w0, b0, w1, b1, x, y, mask = wdata
    mb_losses = []
    for mb in range(_M):
        sl = slice(mb * (_BATCH // _M), (mb + 1) * (_BATCH // _M))
        mb_losses.append(_mb_weighted_loss(w0, b0, w1, b1, x[sl], y[sl], mask[sl]))
    mean_of_means = sum(mb_losses) / _M
    global_mean = _global_weighted_loss(*wdata)
    assert not jnp.allclose(mean_of_means, global_mean, atol=1e-4), "test data must separate the estimators"

    pipe_loss = _make_scheduled(mesh, Std1F1B(microbatches=_M))
    loss, _grads = sxvalue_and_grad(pipe_loss, argnums=(0, 1, 2, 3))(*wdata)
    assert jnp.allclose(loss, mean_of_means, atol=1e-5, rtol=1e-5), (loss, mean_of_means)


def test_unweighted_grads_are_uniform_mean_of_microbatch_grads(mesh, wdata):
    """Uniform grads == mean over microbatches of per-microbatch grads."""
    w0, b0, w1, b1, x, y, mask = wdata

    def mb_grad(mb):
        sl = slice(mb * (_BATCH // _M), (mb + 1) * (_BATCH // _M))
        return jax.grad(_mb_weighted_loss, argnums=(0, 1, 2, 3))(w0, b0, w1, b1, x[sl], y[sl], mask[sl])

    per_mb = [mb_grad(mb) for mb in range(_M)]
    ref_grads = jax.tree.map(lambda *gs: sum(gs) / _M, *per_mb)

    pipe_loss = _make_scheduled(mesh, Std1F1B(microbatches=_M))
    _loss, grads = sxvalue_and_grad(pipe_loss, argnums=(0, 1, 2, 3))(*wdata)
    for got, ref in zip(grads, ref_grads, strict=True):
        assert jnp.allclose(got, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("schedule_cls", [GPipe, Std1F1B])
def test_scheduled_aux_outputs_reduce_with_loss_weights(mesh, wdata, schedule_cls):
    """has_aux returns terminal-stage aux reduced with the same scales as the loss."""
    mask = wdata[6]
    pipe_loss = _make_scheduled(mesh, schedule_cls(microbatches=_M), with_aux=True)
    (loss, aux), grads = sxvalue_and_grad(
        pipe_loss,
        argnums=(0, 1, 2, 3),
        has_aux=True,
        microbatch_weight_fn=_weight_fn,
    )(*wdata)

    ref_loss, ref_grads = jax.value_and_grad(_global_weighted_loss, argnums=(0, 1, 2, 3))(*wdata)
    assert jnp.allclose(loss, ref_loss, atol=1e-5, rtol=1e-5)
    # sq_err aux mirrors the loss, so its weighted reduction must equal the global mean too.
    assert jnp.allclose(aux["sq_err"], ref_loss, atol=1e-5, rtol=1e-5)
    # count reduces to sum(w_mb * w_mb) / sum(w) for the token-count aux.
    counts = [float(jnp.sum(mask[mb * (_BATCH // _M) : (mb + 1) * (_BATCH // _M)])) for mb in range(_M)]
    ref_count = sum(c * c for c in counts) / sum(counts)
    assert jnp.allclose(aux["count"], ref_count, atol=1e-5)
    for got, ref in zip(grads, ref_grads, strict=True):
        assert jnp.allclose(got, ref, atol=1e-5, rtol=1e-5)


def test_has_aux_flag_mismatch_raises(mesh, wdata):
    """has_aux must match the traced function's outputs."""
    plain = _make_scheduled(mesh, Std1F1B(microbatches=_M))
    with pytest.raises(ValueError, match="has_aux=True"):
        sxvalue_and_grad(plain, argnums=(0, 1, 2, 3), has_aux=True)(*wdata)
    with_aux = _make_scheduled(mesh, Std1F1B(microbatches=_M), with_aux=True)
    with pytest.raises(ValueError, match="auxiliary outputs"):
        sxvalue_and_grad(with_aux, argnums=(0, 1, 2, 3))(*wdata)


def test_schedule_units_cached_across_steps(mesh, wdata):
    """The unit DAG is derived once per plan, not once per training step."""
    pipe_loss = _make_scheduled(mesh, Std1F1B(microbatches=_M))
    vg = sxvalue_and_grad(pipe_loss, argnums=(0, 1, 2, 3))
    vg(*wdata)
    plan = pipe_loss._mpmd_state["schedule_plan"]
    cached_first = plan.get("_schedule_units_cache")
    assert cached_first is not None
    preflight_first = plan.get("last_schedule_preflight_stats")
    assert preflight_first is not None
    vg(*wdata)
    assert plan.get("_schedule_units_cache") is cached_first
    assert plan.get("last_schedule_preflight_stats") is preflight_first


def test_warm_compile_bypass_not_entered_on_repeat_dispatch(mesh, wdata, monkeypatch):
    """Steady-state dispatch must not touch process-global JAX config.

    The persistent-cache bypass scope flips ``jax_enable_compilation_cache``
    and ``jax_compilation_cache_dir`` and resets JAX's private cache state.
    That is acceptable exactly once per new schedule signature (warm compile);
    entering it on every dispatch would make concurrent compiles elsewhere in
    the process silently lose persistent caching.
    """
    from spectrax.runtime.mpmd import runtime as mpmd_runtime

    real_bypass = mpmd_runtime._stage_persistent_cache_bypass
    enter_count = {"value": 0}

    def counting_bypass():
        enter_count["value"] += 1
        return real_bypass()

    monkeypatch.setattr(mpmd_runtime, "_stage_persistent_cache_bypass", counting_bypass)

    pipe_loss = _make_scheduled(mesh, Std1F1B(microbatches=_M))
    vg = sxvalue_and_grad(pipe_loss, argnums=(0, 1, 2, 3))
    vg(*wdata)
    first_dispatch_entries = enter_count["value"]
    assert first_dispatch_entries >= 1, "first dispatch must warm-compile inside the bypass scope"

    cache_enabled_before = bool(jax.config.jax_enable_compilation_cache)
    cache_dir_before = jax.config.jax_compilation_cache_dir
    vg(*wdata)
    vg(*wdata)
    assert enter_count["value"] == first_dispatch_entries, (
        "repeat dispatch of an already-warmed schedule entered the persistent-cache bypass scope"
    )
    assert bool(jax.config.jax_enable_compilation_cache) == cache_enabled_before
    assert jax.config.jax_compilation_cache_dir == cache_dir_before


def test_aux_outputs_require_explicit_two_tuple(mesh, wdata):
    """Aux-returning scheduled functions must return (loss, aux), never a flat pytree."""

    @sxjit(mesh=mesh, schedule=Std1F1B(microbatches=_M), batch_argnums=(4, 5, 6))
    def flat_triple(w0, b0, w1, b1, x, y, mask):
        """Returns a flat 3-tuple instead of (loss, aux)."""
        h = jnp.maximum(x @ w0 + b0, 0)
        h = sxstage_iter(h)
        out = h @ w1 + b1
        loss = jnp.sum(((out - y) ** 2) * mask) / jnp.sum(mask)
        return loss, loss * 2.0, jnp.sum(mask)

    with pytest.raises(ValueError, match="explicit 2-tuple"):
        sxvalue_and_grad(flat_triple, argnums=(0, 1, 2, 3), has_aux=True)(*wdata)

    @sxjit(mesh=mesh, schedule=Std1F1B(microbatches=_M), batch_argnums=(4, 5, 6))
    def dict_return(w0, b0, w1, b1, x, y, mask):
        """Returns a dict whose first flattened scalar would silently become the loss."""
        h = jnp.maximum(x @ w0 + b0, 0)
        h = sxstage_iter(h)
        out = h @ w1 + b1
        loss = jnp.sum(((out - y) ** 2) * mask) / jnp.sum(mask)
        return {"loss": loss, "count": jnp.sum(mask)}

    with pytest.raises(ValueError, match="explicit 2-tuple"):
        sxvalue_and_grad(dict_return, argnums=(0, 1, 2, 3), has_aux=True)(*wdata)


def test_aux_two_tuple_with_multi_leaf_loss_element_rejected(mesh, wdata):
    """The first element of (loss, aux) must be a single scalar leaf."""

    @sxjit(mesh=mesh, schedule=Std1F1B(microbatches=_M), batch_argnums=(4, 5, 6))
    def tuple_loss(w0, b0, w1, b1, x, y, mask):
        """First element is itself a 2-leaf pytree."""
        h = jnp.maximum(x @ w0 + b0, 0)
        h = sxstage_iter(h)
        out = h @ w1 + b1
        loss = jnp.sum(((out - y) ** 2) * mask) / jnp.sum(mask)
        return (loss, loss * 2.0), jnp.sum(mask)

    with pytest.raises(ValueError, match="single scalar per-microbatch loss"):
        sxvalue_and_grad(tuple_loss, argnums=(0, 1, 2, 3), has_aux=True)(*wdata)
