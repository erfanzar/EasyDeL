# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for training with narrow-float parameter storage."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from spectrax.quantization import (
    narrow_storage_update,
    scale_loss_for_narrow_gradients,
    stochastic_round,
    suggested_loss_scale,
)

_FP8 = jnp.float8_e4m3b11fnuz


def test_stochastic_rounding_is_unbiased_where_nearest_is_not():
    """A value between two representable neighbours must survive in expectation.

    Round-to-nearest sends every element of a sub-spacing update to the
    same neighbour, so the update is lost identically on every step.
    Stochastic rounding splits them in proportion, and the mean tracks the
    original value.
    """
    target = jnp.full((100_000,), 1.0 + float(jnp.finfo(_FP8).eps) * 0.25, jnp.float32)

    nearest = target.astype(_FP8).astype(jnp.float32)
    assert float(jnp.mean(nearest)) == 1.0, "the offset should vanish under round-to-nearest"

    stochastic = stochastic_round(target, _FP8, jax.random.key(0)).astype(jnp.float32)
    assert 1.0 < float(jnp.mean(stochastic)) < 1.0 + float(jnp.finfo(_FP8).eps)


def test_stochastic_rounding_output_dtype_and_shape():
    """The cast must actually produce the narrow dtype."""
    x = jax.random.normal(jax.random.key(1), (64, 32), jnp.float32)
    out = stochastic_round(x, _FP8, jax.random.key(2))
    assert out.dtype == _FP8
    assert out.shape == x.shape


def test_stochastic_rounding_varies_with_the_key():
    """A fixed key would reintroduce exactly the bias this is meant to remove."""
    x = jnp.full((4096,), 1.0 + float(jnp.finfo(_FP8).eps) * 0.5, jnp.float32)
    first = stochastic_round(x, _FP8, jax.random.key(0))
    second = stochastic_round(x, _FP8, jax.random.key(1))
    assert not jnp.array_equal(first, second)


def test_integer_storage_is_refused_with_the_reason():
    """Integer parameters have no gradient at all, so rounding cannot help them."""
    x = jax.random.normal(jax.random.key(1), (8, 8), jnp.float32)
    with pytest.raises(ValueError, match="no gradient"):
        stochastic_round(x, jnp.int8, jax.random.key(2))


@pytest.mark.parametrize("dtype", [_FP8, jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.bfloat16])
def test_suggested_loss_scale_stays_within_the_dtype(dtype):
    """The scale must not itself push typical gradients past the dtype's maximum.

    Sizing the scale off the underflow floor alone overshoots badly on the
    narrow-exponent types — for ``float8_e4m3b11fnuz``, whose largest finite
    value is 30, it suggested 8192.
    """
    scale = suggested_loss_scale(dtype)
    assert scale >= 1.0
    assert scale <= 32768.0
    assert scale <= float(jnp.finfo(dtype).max) * 1024, "scale is disproportionate to the dtype's range"


def test_loss_scaling_wrapper_multiplies_the_loss():
    """The factor has to be applied before the backward pass, hence to the loss."""
    scaled = scale_loss_for_narrow_gradients(lambda x: (x**2).sum(), 128.0)
    x = jnp.ones((4,), jnp.float32)
    assert float(scaled(x)) == pytest.approx(4.0 * 128.0)
    assert float(jax.grad(scaled)(x)[0]) == pytest.approx(2.0 * 128.0)


def test_narrow_storage_update_keeps_parameters_narrow_and_finite():
    """The step must return the storage dtype, not the wide compute dtype."""
    params = {"w": jax.random.normal(jax.random.key(0), (32, 16), jnp.float32).astype(_FP8)}
    tx = optax.adam(1e-3)
    opt_state = tx.init(jax.tree.map(lambda p: p.astype(jnp.float32), params))
    grads = {"w": jax.random.normal(jax.random.key(1), (32, 16), jnp.float32).astype(_FP8)}

    new_params, _ = narrow_storage_update(params, grads, opt_state, tx, key=jax.random.key(2))

    assert new_params["w"].dtype == _FP8
    assert bool(jnp.all(jnp.isfinite(new_params["w"].astype(jnp.float32))))
    assert not jnp.array_equal(new_params["w"], params["w"]), "no weight moved"


def test_narrow_storage_update_undoes_the_loss_scale():
    """A scaled gradient must produce the same step as an unscaled one."""
    params = {"w": jnp.full((8, 8), 0.5, jnp.float32).astype(_FP8)}
    raw = {"w": jnp.full((8, 8), 0.01, jnp.float32)}
    scale = 256.0

    tx = optax.sgd(0.1)
    wide = jax.tree.map(lambda p: p.astype(jnp.float32), params)

    unscaled, _ = narrow_storage_update(params, raw, tx.init(wide), tx, loss_scale=1.0)
    scaled_grads = jax.tree.map(lambda g: g * scale, raw)
    rescaled, _ = narrow_storage_update(params, scaled_grads, tx.init(wide), tx, loss_scale=scale)

    np.testing.assert_allclose(
        np.asarray(unscaled["w"].astype(jnp.float32)), np.asarray(rescaled["w"].astype(jnp.float32))
    )


def test_adam_moments_must_be_wide_or_the_step_diverges():
    """The reason moments are kept wide: fp8 second moments produce NaN.

    Adam's second moment is a square, so it underflows harder than the
    gradient did, and ``1 / sqrt(0)`` is what blows up. This pins the
    failure so the wide-moment requirement is not quietly dropped.
    """
    params = {"w": jnp.full((64, 64), 0.1, jnp.float32).astype(_FP8)}
    grads = {"w": jnp.full((64, 64), 1e-3, jnp.float32).astype(_FP8)}
    tx = optax.adam(1e-3)

    narrow_state = tx.init(params)  # moments in fp8 -- the wrong way
    broken = params
    for _ in range(10):
        broken, narrow_state = narrow_storage_update(
            broken, grads, narrow_state, tx, compute_dtype=_FP8
        )

    wide_state = tx.init(jax.tree.map(lambda p: p.astype(jnp.float32), params))
    healthy = params
    for _ in range(10):
        healthy, wide_state = narrow_storage_update(healthy, grads, wide_state, tx, compute_dtype=jnp.float32)

    assert bool(jnp.all(jnp.isfinite(healthy["w"].astype(jnp.float32)))), "wide moments should stay finite"
    assert not bool(jnp.all(jnp.isfinite(broken["w"].astype(jnp.float32)))), (
        "fp8 moments were expected to diverge; if this now passes the wide-moment guidance can be relaxed"
    )
