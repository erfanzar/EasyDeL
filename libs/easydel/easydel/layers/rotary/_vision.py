# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared learned-position interpolation and axial RoPE for packed vision grids."""

from collections.abc import Callable, Iterable, Sequence

import numpy as np
from jax import Array
from jax import numpy as jnp


def _vision_grid_shapes(grid_thw: Iterable[Sequence[int]], spatial_merge_size: int) -> list[tuple[int, int, int]]:
    """Validate concrete grid metadata before constructing any position arrays.

    Args:
        grid_thw: Concrete temporal, height and width dimensions per sample.
        spatial_merge_size: Number of adjacent patches merged along each axis.

    Returns:
        Nonempty list of positive grid dimensions divisible by the merge size.

    Raises:
        ValueError: A dimension is nonpositive or a spatial extent cannot be merged.
    """
    if spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be positive")
    grids = [tuple(int(size) for size in grid) for grid in grid_thw]
    if not grids:
        raise ValueError("grid_thw must contain at least one vision grid")
    for grid in grids:
        if len(grid) != 3 or any(size < 1 for size in grid):
            raise ValueError("Each vision grid must contain positive (temporal, height, width) dimensions")
        _, height, width = grid
        if height % spatial_merge_size or width % spatial_merge_size:
            raise ValueError("Vision grid height and width must be divisible by spatial_merge_size")
    return grids


def vision_grid_pos_embed_interpolate(
    pos_embed: Callable[[Array], Array],
    grid_thw: Iterable[Sequence[int]],
    num_grid_per_side: int,
    spatial_merge_size: int,
) -> Array:
    """Bilinearly interpolate learned square-grid positions in spatial-merge order.

    Coordinates span both endpoints of the learned grid (``align_corners=True``).
    Tokens are ordered by merge-block row, merge-block column, intra-block row,
    then intra-block column. Each frame repeats the same spatial positions.
    Keeping the lookup callable outside this helper preserves the caller's
    embedding module, parameter paths and mixed-precision lookup policy.

    Args:
        pos_embed: Embedding lookup accepting flattened integer indices and
            returning ``[num_indices, hidden_size]`` learned positions.
        grid_thw: Concrete/static ``(temporal, height, width)`` rows per sample.
            Traced dimensions are not supported because output sizes vary by grid.
        num_grid_per_side: Side length of the learned square position grid.
        spatial_merge_size: Number of patches per spatial axis in each merge block.

    Returns:
        Interpolated positions of shape ``[sum(t * h * w), hidden_size]`` in
        packed sample/frame order, with the promoted lookup/interpolation dtype.

    Raises:
        ValueError: The learned-grid size or packed grid metadata is invalid.
    """
    grids = _vision_grid_shapes(grid_thw, spatial_merge_size)
    if num_grid_per_side < 1:
        raise ValueError("num_grid_per_side must be positive")
    merge_size = spatial_merge_size
    side = num_grid_per_side
    indices = [[] for _ in range(4)]
    weights = [[] for _ in range(4)]
    for _, height, width in grids:
        h_grid = jnp.linspace(0, side - 1, height)
        w_grid = jnp.linspace(0, side - 1, width)
        h_floor = jnp.floor(h_grid).astype(jnp.int32)
        w_floor = jnp.floor(w_grid).astype(jnp.int32)
        h_ceil = jnp.minimum(h_floor + 1, side - 1)
        w_ceil = jnp.minimum(w_floor + 1, side - 1)
        dh = h_grid - h_floor
        dw = w_grid - w_floor
        corners = (
            (h_floor[:, None] * side + w_floor[None, :], (1 - dh)[:, None] * (1 - dw)[None, :]),
            (h_floor[:, None] * side + w_ceil[None, :], (1 - dh)[:, None] * dw[None, :]),
            (h_ceil[:, None] * side + w_floor[None, :], dh[:, None] * (1 - dw)[None, :]),
            (h_ceil[:, None] * side + w_ceil[None, :], dh[:, None] * dw[None, :]),
        )
        for corner, (idx, weight) in enumerate(corners):
            indices[corner].append(idx.reshape(-1))
            weights[corner].append(weight.reshape(-1))
    # Batch the four lookups across samples, preserving the vectorized vision path.
    weighted = [
        pos_embed(jnp.concatenate(idx)) * jnp.concatenate(weight)[:, None]
        for idx, weight in zip(indices, weights, strict=True)
    ]
    interpolated = weighted[0] + weighted[1] + weighted[2] + weighted[3]
    positions = []
    offset = 0
    for temporal, height, width in grids:
        pos = interpolated[offset : offset + height * width]
        offset += height * width
        pos = pos.reshape(height // merge_size, merge_size, width // merge_size, merge_size, -1)
        pos = jnp.transpose(pos, (0, 2, 1, 3, 4)).reshape(height * width, -1)
        positions.append(jnp.tile(pos, (temporal, 1)))
    return jnp.concatenate(positions, axis=0)


def vision_grid_rotary_embeddings(
    grid_thw: Iterable[Sequence[int]],
    max_grid_size: int,
    head_dim: int,
    spatial_merge_size: int,
    *,
    theta: float = 10000.0,
) -> Array:
    """Compute height/width RoPE angles in the packed spatial-merge token order.

    Args:
        grid_thw: Concrete/static ``(temporal, height, width)`` rows per sample.
        max_grid_size: Frequency-table extent, at least the largest spatial size.
        head_dim: Even per-axis rotary dimension, e.g. the vision encoder's
            existing ``_head_dim_ro`` (not its full attention head dimension).
        spatial_merge_size: Number of patches per spatial axis in each merge block.
        theta: Base period for inverse frequencies.

    Returns:
        Float32 angles ``[sum(t * h * w), head_dim]``: height frequencies followed
        by width frequencies. Callers retain their existing duplication/cos/sin step.

    Raises:
        ValueError: Grid metadata, rotary dimension, base or table extent is invalid.
    """
    grids = _vision_grid_shapes(grid_thw, spatial_merge_size)
    if head_dim < 2 or head_dim % 2:
        raise ValueError("head_dim must be a positive even per-axis rotary dimension")
    if max_grid_size < max(max(height, width) for _, height, width in grids):
        raise ValueError("max_grid_size must cover every vision grid's height and width")
    if theta <= 0:
        raise ValueError("theta must be positive")
    merge_size = spatial_merge_size
    pos_ids = []
    for temporal, height, width in grids:
        h_ids = jnp.broadcast_to(jnp.arange(height)[:, None], (height, width))
        w_ids = jnp.broadcast_to(jnp.arange(width)[None, :], (height, width))
        shape = (height // merge_size, merge_size, width // merge_size, merge_size)
        h_ids = h_ids.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
        w_ids = w_ids.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
        pos_ids.append(jnp.tile(jnp.stack([h_ids, w_ids], axis=-1), (temporal, 1)))
    pos_ids = jnp.concatenate(pos_ids, axis=0)
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freq_table = jnp.outer(jnp.arange(max_grid_size, dtype=jnp.float32), inv_freq)
    return freq_table[pos_ids].reshape(pos_ids.shape[0], -1)


def vision_grid_position_ids(
    grid_thw: Iterable[Sequence[int]] | np.ndarray,
    spatial_merge_size: int,
) -> np.ndarray:
    """Compute block-major 2-D ``(row, col)`` patch position ids for packed vision grids.

    Tokens within each frame are ordered by merge-block row, merge-block column,
    intra-block row, then intra-block column, and repeated ``temporal`` times.

    Args:
        grid_thw: Concrete/static ``(temporal, height, width)`` rows per sample.
        spatial_merge_size: Number of adjacent patches merged along each axis.

    Returns:
        NumPy ``int64`` array of shape ``[sum(t * h * w), 2]`` holding
        ``(row, col)`` patch coordinates.
    """
    grids = _vision_grid_shapes(grid_thw, spatial_merge_size)
    merge = int(spatial_merge_size)
    pos_list: list[np.ndarray] = []
    for temporal, height, width in grids:
        h_grid, w_grid = np.meshgrid(np.arange(height, dtype=np.int64), np.arange(width, dtype=np.int64), indexing="ij")
        block_shape = (height // merge, merge, width // merge, merge)
        h_flat = h_grid.reshape(block_shape).transpose(0, 2, 1, 3).reshape(-1)
        w_flat = w_grid.reshape(block_shape).transpose(0, 2, 1, 3).reshape(-1)
        frame_pos = np.stack([h_flat, w_flat], axis=-1)
        pos_list.append(np.tile(frame_pos, (temporal, 1)))
    return np.concatenate(pos_list, axis=0)


def _bilinear_axis_taps_and_weights(
    index: np.ndarray,
    size: np.ndarray,
    side: int,
    *,
    align_corners: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1-D bilinear interpolation indices and weights matching ``F.interpolate``."""
    idx = index.astype(np.float32)
    sz = size.astype(np.float32)
    if align_corners:
        src = idx * (side - 1) / np.maximum(sz - 1.0, 1.0)
    else:
        src = (idx + 0.5) * side / sz - 0.5
    floor = np.floor(src)
    offsets = np.array([0, 1], dtype=np.int64)
    raw_taps = floor.astype(np.int64)[:, None] + offsets[None, :]
    taps = np.clip(raw_taps, 0, side - 1)
    distance = np.abs(src[:, None] - floor[:, None] - offsets[None, :].astype(np.float32))
    weights = np.maximum(1.0 - distance, 0.0).astype(np.float32)
    return taps, weights


def vision_grid_pos_embed_interpolation_indices_and_weights(
    grid_thw: Iterable[Sequence[int]] | np.ndarray,
    num_grid_per_side: int,
    *,
    spatial_merge_size: int = 2,
    align_corners: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 4-tap bilinear lookup indices and weights for a square learned position grid.

    Supports both ``align_corners=False`` (pixel-center resampling matching
    PyTorch's default ``F.interpolate(..., mode="bilinear", align_corners=False)``,
    used by Inkling 2) and ``align_corners=True``.

    Args:
        grid_thw: Concrete/static ``(temporal, height, width)`` rows per sample.
        num_grid_per_side: Side length of the learned square position grid.
        spatial_merge_size: Number of patches per spatial axis in each merge block.
        align_corners: Whether grid corners align with endpoints.

    Returns:
        Tuple ``(indices, weights)`` where ``indices`` is ``int64`` of shape
        ``[sum(t * h * w), 4]`` into the flattened ``side * side`` table and
        ``weights`` is ``float32`` of shape ``[sum(t * h * w), 4]``.
    """
    grids = _vision_grid_shapes(grid_thw, spatial_merge_size)
    side = int(num_grid_per_side)
    if side < 1:
        raise ValueError("num_grid_per_side must be positive")
    merge = int(spatial_merge_size)
    grid_np = np.asarray(grids, dtype=np.int64)

    counts = grid_np[:, 0] * grid_np[:, 1] * grid_np[:, 2]
    heights = np.repeat(grid_np[:, 1], counts)
    widths = np.repeat(grid_np[:, 2], counts)
    starts_per_img = np.concatenate([np.zeros((1,), dtype=np.int64), np.cumsum(counts)[:-1]])
    starts = np.repeat(starts_per_img, counts)

    within = (np.arange(int(counts.sum()), dtype=np.int64) - starts) % (heights * widths)
    blocks_w = widths // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    row = block_row * merge + in_row
    col = block_col * merge + in_col

    h_taps, h_weights = _bilinear_axis_taps_and_weights(row, heights, side, align_corners=align_corners)
    w_taps, w_weights = _bilinear_axis_taps_and_weights(col, widths, side, align_corners=align_corners)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    return indices, weights
