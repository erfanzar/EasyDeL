# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import importlib
from types import SimpleNamespace

import jax
from jax.sharding import NamedSharding, PartitionSpec

stage_mesh = importlib.import_module("easydel.infra.sharding")


def test_resolve_stage_mesh_prefers_spectrax_stage_mesh(monkeypatch):
    calls = []

    def get_current_stage_mesh(mesh=None, *, arr=None, stage=None, raise_error=False):
        calls.append((mesh, arr, stage, raise_error))
        return SimpleNamespace(shape={"x": 1}, empty=False)

    monkeypatch.setattr(stage_mesh.spx, "get_current_stage_mesh", get_current_stage_mesh)
    resolved = stage_mesh.resolve_stage_mesh("global", arr="arr", stage=(1, 2))

    assert resolved.shape == {"x": 1}
    assert calls == [("global", "arr", (1, 2), False)]


def test_resolve_stage_mesh_falls_back_to_plain_mesh(monkeypatch):
    plain = SimpleNamespace(shape={"x": 1}, empty=False)

    def get_current_stage_mesh(*args, **kwargs):
        return None

    monkeypatch.setattr(stage_mesh.spx, "get_current_stage_mesh", get_current_stage_mesh)
    monkeypatch.setattr(stage_mesh.spx, "to_jax_mesh", lambda mesh: plain)

    assert stage_mesh.resolve_stage_mesh("global") is plain


def test_resolve_array_mesh_uses_named_sharding_mesh():
    mesh = jax.sharding.Mesh(jax.devices()[:1], ("x",))
    arr = jax.device_put(jax.numpy.ones((1,)), NamedSharding(mesh, PartitionSpec()))

    assert stage_mesh.resolve_array_mesh(arr) is mesh
