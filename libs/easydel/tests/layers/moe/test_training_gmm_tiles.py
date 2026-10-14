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


@pytest.mark.parametrize("shape", [GATE_UP, DOWN], ids=["gate_up", "down"])
def test_split_gmm_forward_is_xla_and_gradients_match(shape):
    """XLA forward + Pallas backward: the forward equals the XLA grouped matmul bit for bit (eval traces are
    unchanged), and both gradients match the all-XLA vjp within bf16 tolerance."""
    import jax
    import numpy as np
    from easydel.layers.moe._moe_module import _xla_forward_pallas_backward_gmm
    from ejkernel.modules import GroupedMatmulConfig, grouped_matmul

    if jax.default_backend() != "tpu":
        pytest.skip("Pallas grouped matmul needs TPU")
    experts, k, n = shape
    rows = 1536
    rng = np.random.default_rng(3)
    sizes = jnp.asarray(rng.multinomial(rows, np.full(experts, 1.0 / experts)).astype(np.int32))
    x = jnp.asarray(rng.standard_normal((rows, k)) * 0.1, jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((experts, k, n)) * 0.02, jnp.bfloat16)
    ct = jnp.asarray(rng.standard_normal((rows, n)), jnp.bfloat16)
    xla_kws = {
        "preferred_element_type": jnp.bfloat16,
        "cfg": GroupedMatmulConfig(platform="xla", bypass_xla_tiling=True),
    }
    bm, bk, bn = _training_gmm_tiles(rows, shape, jnp.bfloat16)
    pallas_cfg = GroupedMatmulConfig(platform="pallas", bypass_xla_tiling=True, block_m=bm, block_k=bk, block_n=bn)
    pallas_kws = dict(xla_kws, cfg=pallas_cfg)

    def split(a, b, g):
        return _xla_forward_pallas_backward_gmm(a, b, g, xla_kws=xla_kws, pallas_kws=pallas_kws)

    def plain(a, b, g):
        return grouped_matmul(a, b, g, **xla_kws)

    def fwdbwd(f):
        # Arrays are jit arguments, not closure constants (closing over them bakes ~2 GB into the program).
        def run(a, b, g, c):
            out, vjp = jax.vjp(lambda p, q: f(p, q, g), a, b)
            return out, *vjp(c)

        return jax.jit(run)(x, w, sizes, ct)

    got = fwdbwd(split)
    ref = fwdbwd(plain)
    np.testing.assert_array_equal(np.asarray(jax.jit(split)(x, w, sizes)), np.asarray(jax.jit(plain)(x, w, sizes)))
    np.testing.assert_array_equal(np.asarray(got[0]), np.asarray(ref[0]))
    for name, a, b in (("dlhs", got[1], ref[1]), ("drhs", got[2], ref[2])):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < 1e-4, name
