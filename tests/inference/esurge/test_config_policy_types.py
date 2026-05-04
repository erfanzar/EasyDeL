# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import pytest

from easydel.inference.esurge.config import (
    eSurgeRuntimeConfig,
    normalize_kernel_tile_policy,
)
from easydel.inference.esurge.runners import pipeline_plan as pp_plan


class _FakeTextConfig:
    num_hidden_layers = 4


class _FakeConfig:
    @staticmethod
    def get_text_config():
        return _FakeTextConfig()


class _FakeModel:
    config = _FakeConfig()

    def __init__(self, *, mesh) -> None:
        self.mesh = mesh

    @staticmethod
    def get_operations_cache_view():
        return {}


class _FakeMpmdMesh:
    mpmd_dim = 2

    @staticmethod
    def submesh(rank: int):
        return f"stage-{rank}"


class _FakeMesh:
    mpmd_mesh = _FakeMpmdMesh()


def test_kernel_tile_policy_normalizer():
    assert normalize_kernel_tile_policy(None) == "auto"
    assert normalize_kernel_tile_policy("B8") == "b8"

    with pytest.raises(ValueError, match="kernel_tile_policy"):
        normalize_kernel_tile_policy("b32")


def test_runtime_defaults_stay_library_defaults():
    runtime = eSurgeRuntimeConfig.from_dict()

    assert runtime.min_input_pad == 16
    assert runtime.max_num_seqs == 256
    assert runtime.max_num_seq_buckets is None
    assert runtime.pp_microbatch_count == "auto"
    assert runtime.pp_microbatch_size == "auto"


def test_pipeline_plan_is_derived_from_mesh(monkeypatch):
    monkeypatch.setattr(pp_plan, "is_mpmd_mesh", lambda mesh: isinstance(mesh, _FakeMesh))

    spmd_plan = pp_plan.build_pipeline_inference_plan(model=_FakeModel(mesh=object()))
    mpmd_plan = pp_plan.build_pipeline_inference_plan(model=_FakeModel(mesh=_FakeMesh()))

    assert spmd_plan.enabled is False
    assert mpmd_plan.enabled is True
    assert mpmd_plan.mpmd_dim == 2
    assert mpmd_plan.stage_to_layers == {0: (0, 1), 1: (2, 3)}
