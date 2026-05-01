# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import pytest

from easydel.inference.esurge.config import (
    normalize_kernel_tile_policy,
    normalize_pipeline_inference_mode,
)


def test_pipeline_inference_mode_normalizer():
    assert normalize_pipeline_inference_mode(None) == "auto"
    assert normalize_pipeline_inference_mode("ON") == "on"

    with pytest.raises(ValueError, match="pipeline_inference"):
        normalize_pipeline_inference_mode("maybe")


def test_kernel_tile_policy_normalizer():
    assert normalize_kernel_tile_policy(None) == "auto"
    assert normalize_kernel_tile_policy("B8") == "b8"

    with pytest.raises(ValueError, match="kernel_tile_policy"):
        normalize_kernel_tile_policy("b32")
