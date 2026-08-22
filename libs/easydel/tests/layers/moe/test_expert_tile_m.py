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

"""The stacked-expert grouped matmul must size its m-tile by rows-per-expert.

Each non-empty expert costs one full m-tile however few rows it holds, so the
tile is set by an expert's row count, not by the height of the permuted buffer.
Those differ by orders of magnitude once there are many experts, and the old
buffer-height rule therefore bought 128-row tiles for 48-row groups.

Measured on v5p against the tile this picks: 1.77x at 2048-token prefill,
1.64x at 32-token decode, 1.53x at 8-token decode (DeepSeek-V4 shapes, int4).
"""

import pytest
from easydel.layers.moe._moe_module import _expert_tile_m


def _old_rule(buffer_rows: int) -> int:
    """The superseded buffer-height rule, kept to show where they differ."""
    return 64 if buffer_rows <= 1024 else 128


@pytest.mark.parametrize(
    ("tokens", "top_k", "num_experts", "expected"),
    [
        (2048, 6, 256, 64),  # DeepSeek-V4 prefill: 48 rows/expert -> measured optimum 64
        (32, 6, 256, 8),  # DeepSeek-V4 decode cc32: 0.75 rows/expert -> measured optimum 8
        (8, 6, 256, 8),  # tiny decode batch: 0.19 rows/expert -> measured optimum 8
    ],
)
def test_matches_the_measured_optimum(tokens, top_k, num_experts, expected):
    assert _expert_tile_m(tokens * top_k, num_experts) == expected


@pytest.mark.parametrize(
    ("tokens", "top_k", "num_experts"),
    [
        (2048, 2, 8),  # mixtral-style: 512 rows/expert
        (2048, 8, 128),  # qwen3-moe-style: 128 rows/expert
    ],
)
def test_few_expert_models_keep_the_large_tile(tokens, top_k, num_experts):
    """Models whose experts already fill a tile must not be perturbed.

    The change has to be a strict improvement or a no-op per family, otherwise
    it trades one model's regression for another's win.
    """
    buffer_rows = tokens * top_k
    assert _expert_tile_m(buffer_rows, num_experts) == 128 == _old_rule(buffer_rows)


def test_result_is_always_a_power_of_two_within_mosaic_bounds():
    """Mosaic wants 8-aligned tiles; 128 is the largest that ever helped."""
    for buffer_rows in (1, 7, 48, 192, 3072, 12288, 1 << 20):
        for num_experts in (1, 8, 64, 128, 256, 1024):
            tile = _expert_tile_m(buffer_rows, num_experts)
            assert 8 <= tile <= 128
            assert tile & (tile - 1) == 0, f"{tile} is not a power of two"
            assert tile % 8 == 0


def test_tile_is_monotone_in_rows_per_expert():
    """More rows per expert must never ask for a smaller tile."""
    tiles = [_expert_tile_m(rows, 256) for rows in (48, 192, 3072, 12288, 65536)]
    assert tiles == sorted(tiles)


def test_never_smaller_than_the_rows_it_must_hold_until_clamped():
    """The tile should cover an average expert's rows until it hits the cap."""
    for num_experts in (64, 256):
        for buffer_rows in (num_experts, num_experts * 3, num_experts * 40):
            rows_per_expert = -(-buffer_rows // num_experts)
            tile = _expert_tile_m(buffer_rows, num_experts)
            if tile < 128:
                assert tile >= rows_per_expert, f"tile {tile} under-covers {rows_per_expert} rows"


def test_zero_experts_does_not_divide_by_zero():
    assert _expert_tile_m(1024, 0) == 128
