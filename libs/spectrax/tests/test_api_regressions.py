# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the public ``spectrax.run`` API."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import spectrax as spx
from spectrax.api import _place_state, _run_mpmd


class _Identity(spx.Module):
    """Small module for API validation tests."""

    def forward(self, x):
        """Run the forward pass."""
        return x


class _Weighted(spx.Module):
    """Small stateful module for API state-placement regressions."""

    def __init__(self):
        """Initialize with w."""
        super().__init__()
        self.w = spx.Parameter(jnp.asarray(1.0))

    def forward(self, x):
        """Run the forward pass."""
        return x * self.w.value


def test_run_rejects_invalid_mode_before_dispatch():
    """Mode validation should not silently route unknown strings to training."""
    mesh = spx.create_mesh(axis_dims=(-1,), axis_names=("data",))

    with pytest.raises(ValueError, match="mode"):
        spx.run(_Identity(), inputs=jnp.ones((1,)), mesh=mesh, mode="bogus")


def test_run_rejects_targets_in_forward_mode():
    """Forward mode should not silently drop loss targets."""
    mesh = spx.create_mesh(axis_dims=(-1,), axis_names=("data",))

    with pytest.raises(ValueError, match="targets"):
        spx.run(_Identity(), inputs=jnp.ones((1,)), targets=jnp.ones((1,)), mesh=mesh, mode="forward")


def test_mpmd_train_requires_loss_fn_even_without_explicit_schedule():
    """MPMD train must not substitute a dummy zero loss."""
    mesh = spx.create_mesh(axis_dims=(-1,), axis_names=("pp",), mpmd_axis="pp")

    with pytest.raises(ValueError, match="loss_fn"):
        _run_mpmd(
            _Identity(),
            (jnp.ones((1,)),),
            {},
            mesh=mesh,
            mode="train",
            loss_args=(),
            loss_kwargs={},
            loss_fn=None,
            microbatches=1,
        )


def test_mpmd_train_requires_loss_fn_even_with_loss_kwargs():
    """Loss-kwarg wrapping must not hide a missing loss function."""
    mesh = spx.create_mesh(axis_dims=(-1,), axis_names=("pp",), mpmd_axis="pp")

    with pytest.raises(ValueError, match="loss_fn"):
        _run_mpmd(
            _Identity(),
            (jnp.ones((1,)),),
            {},
            mesh=mesh,
            mode="train",
            loss_args=(),
            loss_kwargs={"labels": jnp.ones((1,))},
            loss_fn=None,
            microbatches=1,
        )


def test_mpmd_requires_positional_input_batch():
    """Direct MPMD callers should get a clear error before indexing ``args[0]``."""
    mesh = spx.create_mesh(axis_dims=(-1,), axis_names=("pp",), mpmd_axis="pp")

    with pytest.raises(ValueError, match="positional input batch"):
        _run_mpmd(
            _Identity(),
            (),
            {},
            mesh=mesh,
            mode="forward",
            loss_args=(),
            loss_kwargs={},
            loss_fn=None,
            microbatches=1,
        )


def test_mpmd_default_schedule_forwards_runtime_flags(monkeypatch):
    """The implicit GPipe path should preserve the same flags as explicit schedules."""
    mesh = spx.create_mesh(axis_dims=(-1,), axis_names=("pp",), mpmd_axis="pp")
    seen = {}

    def fake_sxcall(*_args, **kwargs):
        """Fake sxcall for monkeypatching."""
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr("spectrax.api.sxcall", fake_sxcall)

    out = _run_mpmd(
        _Identity(),
        (jnp.ones((1,)),),
        {},
        mesh=mesh,
        mode="train",
        loss_args=(jnp.ones((1,)),),
        loss_kwargs={},
        loss_fn=lambda y, t: ((y - t) ** 2).mean(),
        microbatches=2,
        fuse_1f1b=True,
        fuse_zb=True,
        has_aux=True,
    )

    assert out == "ok"
    assert seen["fuse_1f1b"] is True
    assert seen["fuse_zb"] is True
    assert seen["has_aux"] is True


def test_place_state_preserves_live_writers():
    """Placed SPMD state should keep writer callbacks to live variables."""
    model = _Weighted()
    _gdef, state = spx.export(model)
    mesh = spx.create_mesh(axis_dims=(-1,), axis_names=("data",)).jax_mesh

    placed = _place_state(state, model, mesh)
    placed.set("parameters", "w", jnp.asarray(3.0))

    assert float(model.w.value) == 3.0
