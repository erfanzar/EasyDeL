# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.nn.norm`."""

from __future__ import annotations

import jax.numpy as jnp

from spectrax.nn.norm import BatchNorm1d, BatchNorm2d, LayerNorm, RMSNorm


def test_layernorm_identity_on_zero_variance_affine():
    """With affine defaults (scale=1, bias=0), a constant input yields zeros."""
    m = LayerNorm(4)
    x = jnp.full((2, 4), 3.0)
    y = m(x)
    assert jnp.allclose(y, 0.0, atol=1e-4)


def test_layernorm_unit_variance():
    """Output has (approximately) unit variance along the last axis."""
    m = LayerNorm(8)
    rng_state = jnp.arange(16.0).reshape(2, 8)
    y = m(rng_state)
    assert jnp.allclose(jnp.var(y, axis=-1), 1.0, atol=1e-3)


def test_layernorm_affine_off_has_no_scale():
    """Without affine the layer has no scale/bias parameters."""
    m = LayerNorm(4, elementwise_affine=False)
    assert not hasattr(m, "weight")
    assert not hasattr(m, "bias")


def test_layernorm_use_bias_false():
    """``use_bias=False`` keeps scale but drops bias."""
    m = LayerNorm(4, use_bias=False)
    assert hasattr(m, "weight")
    assert not hasattr(m, "bias")


def test_layernorm_scale_applied():
    """Setting ``scale=2`` doubles the normalized output."""
    m = LayerNorm(4, use_bias=False)
    m.weight.value = jnp.full((4,), 2.0)
    x = jnp.arange(8.0).reshape(2, 4)
    y = m(x)
    x_norm_var = jnp.var(y, axis=-1)
    assert jnp.allclose(x_norm_var, 4.0, atol=1e-3)


def test_rmsnorm_forward_shape_and_scale():
    """RMSNorm produces same-shape output with a per-feature scale."""
    m = RMSNorm(4)
    x = jnp.arange(8.0).reshape(2, 4)
    assert m(x).shape == x.shape


def test_rmsnorm_without_affine():
    """Without affine, RMSNorm has no parameters."""
    m = RMSNorm(4, elementwise_affine=False)
    assert not hasattr(m, "weight")


def test_rmsnorm_divides_by_rms():
    """Output is ``x / sqrt(mean(x**2) + eps)``."""
    m = RMSNorm(4, elementwise_affine=False)
    x = jnp.asarray([[3.0, 4.0, 0.0, 0.0]])
    y = m(x)
    manual = x / jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + m.eps)
    assert jnp.allclose(y, manual, atol=1e-5)


def test_batchnorm1d_running_stats_initialized():
    """BatchNorm allocates running mean (zeros) and var (ones)."""
    bn = BatchNorm1d(4)
    assert jnp.array_equal(bn.running_mean.value, jnp.zeros(4))
    assert jnp.array_equal(bn.running_var.value, jnp.ones(4))
    assert bn.running_mean.kind == "batch_stats"


def test_batchnorm_training_updates_running_stats():
    """Training mode blends the batch statistics into the running ones."""
    bn = BatchNorm1d(2, momentum=0.5)
    x = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    bn.train()
    _ = bn(x)
    assert not jnp.array_equal(bn.running_mean.value, jnp.zeros(2))


def test_batchnorm_eval_uses_running_stats():
    """Eval mode consumes the running stats without updating them."""
    bn = BatchNorm1d(2)
    bn.running_mean.value = jnp.asarray([1.0, 1.0])
    bn.running_var.value = jnp.asarray([1.0, 1.0])
    bn.eval()
    x = jnp.asarray([[2.0, 3.0]])
    y = bn(x)
    expected = (x - 1.0) / jnp.sqrt(1.0 + bn.eps)
    if bn.affine:
        expected = expected * bn.weight.value + bn.bias.value
    assert jnp.allclose(y, expected, atol=1e-5)


def test_batchnorm_without_affine():
    """``affine=False`` omits scale and bias parameters."""
    bn = BatchNorm1d(4, affine=False)
    assert not hasattr(bn, "weight")
    assert not hasattr(bn, "bias")


def test_batchnorm2d_forward_shape():
    """``BatchNorm2d`` accepts ``(N, H, W, C)`` inputs."""
    bn = BatchNorm2d(3)
    x = jnp.zeros((2, 4, 4, 3))
    assert bn(x).shape == x.shape


def test_batchnorm_eps_is_static_field():
    """``eps`` lands in ``_spx_static``."""
    bn = BatchNorm1d(4, eps=1e-4)
    assert bn._spx_static["eps"] == 1e-4


def test_norm_layers_expose_sharding_overrides():
    """Norm parameters and running stats accept explicit sharding."""
    ln = LayerNorm(4, sharding=("tp",), bias_sharding=("fsdp",))
    rms = RMSNorm(4, sharding=("tp",))
    bn = BatchNorm1d(
        4,
        sharding=("tp",),
        bias_sharding=("fsdp",),
        stats_sharding=("tp",),
    )
    assert ln.weight.sharding is not None
    assert ln.weight.sharding.axis_names == ("tp",)
    assert ln.bias.sharding is not None
    assert ln.bias.sharding.axis_names == ("fsdp",)
    assert rms.weight.sharding is not None
    assert rms.weight.sharding.axis_names == ("tp",)
    assert bn.weight.sharding is not None
    assert bn.weight.sharding.axis_names == ("tp",)
    assert bn.bias.sharding is not None
    assert bn.bias.sharding.axis_names == ("fsdp",)
    assert bn.running_mean.sharding is not None
    assert bn.running_mean.sharding.axis_names == ("tp",)
    assert bn.running_var.sharding is not None
    assert bn.running_var.sharding.axis_names == ("tp",)
