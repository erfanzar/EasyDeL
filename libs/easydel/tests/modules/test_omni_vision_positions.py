"""Independent references for Omni's public packed-vision position methods."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.modules.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeVisionEncoder
from jax import numpy as jnp


@pytest.mark.parametrize("storage_dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("grids,merge", [([(2, 4, 6), (1, 2, 4)], 2), ([(1, 1, 1), (2, 3, 2)], 1)])
def test_omni_vision_positions_keep_interpolation_rope_and_spatial_merge_order(storage_dtype, grids, merge):
    side, head_dim = 3, 4
    stored = jnp.asarray(np.random.default_rng(29).normal(size=(side * side, 5)), dtype=storage_dtype)
    table = np.asarray(stored, dtype=np.float32)
    tower = SimpleNamespace(
        pos_embed=lambda ids: stored[ids],
        num_grid_per_side=side,
        spatial_merge_size=merge,
        _head_dim_ro=head_dim,
    )
    expected_positions, expected_angles = [], []
    # Scalar iteration defines token order independently of reshape/transposes.
    for frames, height, width in grids:
        for _ in range(frames):
            for block_h in range(0, height, merge):
                for block_w in range(0, width, merge):
                    for h in range(block_h, block_h + merge):
                        for w in range(block_w, block_w + merge):
                            row = h * (side - 1) / (height - 1) if height > 1 else 0.0
                            col = w * (side - 1) / (width - 1) if width > 1 else 0.0
                            low_h, low_w = math.floor(row), math.floor(col)
                            high_h, high_w = min(low_h + 1, side - 1), min(low_w + 1, side - 1)
                            dh, dw = row - low_h, col - low_w
                            expected_positions.append(
                                table[low_h * side + low_w] * ((1 - dh) * (1 - dw))
                                + table[low_h * side + high_w] * ((1 - dh) * dw)
                                + table[high_h * side + low_w] * (dh * (1 - dw))
                                + table[high_h * side + high_w] * (dh * dw)
                            )
                            expected_angles.append(
                                [
                                    position / 10000.0 ** (i / head_dim)
                                    for position in (h, w)
                                    for i in range(0, head_dim, 2)
                                ]
                            )
    grid = np.asarray(grids)
    positions = Qwen3OmniMoeVisionEncoder.fast_pos_embed_interpolate(tower, grid)
    angles = Qwen3OmniMoeVisionEncoder.rot_pos_emb(tower, grid, max(max(h, w) for _, h, w in grids))
    assert tower._head_dim_ro == head_dim
    assert positions.shape == (len(expected_positions), 5)
    assert positions.dtype == jnp.float32
    assert angles.shape == (len(expected_angles), head_dim)
    assert angles.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(positions), expected_positions, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(angles), expected_angles, rtol=2e-5, atol=2e-6)
