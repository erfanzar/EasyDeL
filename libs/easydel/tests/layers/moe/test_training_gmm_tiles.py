# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Shape policy for the measured bf16 training grouped-matmul tiles (pure shape logic, no array computation).

The tiles and row ranges come from TPU v5p measurements recorded beside ``_TRAINING_GMM_TILES`` in
``easydel/layers/moe/_moe_module.py``; this test pins the policy boundaries those measurements support.
"""

import jax.numpy as jnp
import pytest
from easydel.layers.moe._moe_module import _training_gmm_tiles

GATE_UP = (96, 5120, 2304)
DOWN = (96, 2304, 5120)


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    ("rows", "shape", "expected"),
    [
        (1536, GATE_UP, (128, 1024, 1152)),
        (12288, GATE_UP, (128, 1024, 1152)),
        (12289, GATE_UP, (256, 1024, 1152)),
        (49152, GATE_UP, (256, 1024, 1152)),
        (1536, DOWN, (128, 1152, 1280)),
        (24576, DOWN, (256, 1152, 1280)),
        (49152, DOWN, (256, 1152, 1280)),
    ],
)
def test_measured_families_get_wide_k_tiles(rows, shape, expected):
    assert _training_gmm_tiles(rows, shape, jnp.bfloat16) == expected


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    ("rows", "shape", "dtype"),
    [
        (1535, GATE_UP, jnp.bfloat16),  # below the measured row range
        (49153, DOWN, jnp.bfloat16),  # above the measured row range
        (4096, (96, 5120, 4608), jnp.bfloat16),  # fused gate_up: never measured
        (4096, (128, 2560, 1280), jnp.bfloat16),  # other expert count
        (4096, GATE_UP, jnp.float32),  # f32 keeps the caller's path
    ],
)
def test_unmeasured_cases_keep_callers_choice(rows, shape, dtype):
    assert _training_gmm_tiles(rows, shape, dtype) is None
