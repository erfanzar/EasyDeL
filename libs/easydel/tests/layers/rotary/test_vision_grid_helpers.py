"""Independent numerical references for shared packed vision positions (accelerator only)."""

import math

import numpy as np
import pytest
from easydel.layers.rotary import vision_grid_pos_embed_interpolate, vision_grid_rotary_embeddings
from jax import numpy as jnp


def _packed_coordinates(grids, merge):
    """Enumerate spatial-merge order independently using scalar loops."""
    return [
        (h, w, height, width)
        for frames, height, width in grids
        for _ in range(frames)
        for block_h in range(0, height, merge)
        for block_w in range(0, width, merge)
        for h in range(block_h, block_h + merge)
        for w in range(block_w, block_w + merge)
    ]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("grids,merge", [([(2, 4, 6), (1, 2, 4)], 2), ([(1, 1, 1), (2, 3, 2)], 1)])
def test_vision_interpolation_matches_scalar_bilinear_reference(dtype, grids, merge):
    side = 3
    stored = jnp.asarray(np.random.default_rng(17).normal(size=(side * side, 5)), dtype=dtype)
    table = np.asarray(stored, dtype=np.float32)
    expected = []
    for h, w, height, width in _packed_coordinates(grids, merge):
        row = h * (side - 1) / (height - 1) if height > 1 else 0.0
        col = w * (side - 1) / (width - 1) if width > 1 else 0.0
        low_h, low_w = math.floor(row), math.floor(col)
        high_h, high_w = min(low_h + 1, side - 1), min(low_w + 1, side - 1)
        dh, dw = row - low_h, col - low_w
        expected.append(
            table[low_h * side + low_w] * ((1 - dh) * (1 - dw))
            + table[low_h * side + high_w] * ((1 - dh) * dw)
            + table[high_h * side + low_w] * (dh * (1 - dw))
            + table[high_h * side + high_w] * (dh * dw)
        )
    result = vision_grid_pos_embed_interpolate(lambda ids: stored[ids], np.asarray(grids), side, merge)
    assert result.shape == (len(expected), 5)
    assert result.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("head_dim,theta", [(4, 10000.0), (8, 500000.0)])
@pytest.mark.parametrize("merge", [1, 2])
def test_vision_rope_matches_scalar_axial_frequency_reference(head_dim, theta, merge):
    grids = [(2, 4, 6), (1, 2, 4)]
    expected = [
        [position / theta ** (index / head_dim) for position in (h, w) for index in range(0, head_dim, 2)]
        for h, w, _, _ in _packed_coordinates(grids, merge)
    ]
    result = vision_grid_rotary_embeddings(np.asarray(grids), 6, head_dim, merge, theta=theta)
    assert result.shape == (len(expected), head_dim)
    assert result.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), rtol=2e-5, atol=2e-6)
